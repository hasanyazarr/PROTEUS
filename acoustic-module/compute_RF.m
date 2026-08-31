function RF = compute_RF(Transducer, sensor_data, sensor_weights, ...
    Grid, run_param)
%COMPUTE_RF Convert pressure sensor data on the transducer to voltage
%element data.
%
% The work is blocked along the transducer element axis. Every array this
% function builds is sized by the transducer's integration points and the
% receive window, and on the GPU each one is a single gpuArray -- which
% MATLAB caps at intmax('int32') elements whatever the device has free.
% v10's lambda/8 grid crosses that cap four times over:
%
%   gpuArray(sensor_data.p)   201795 x 12444 = 2.51e9   1.17x the limit
%   sensor_weights*p          190080 x 12444 = 2.37e9   1.10x
%   fft(p,N,2)                190080 x 12500 = 2.38e9   1.11x
%   exp(-2i*pi*delays*f)      190080 x 12500 = 2.38e9   1.11x
%
% The sensor axis is reduced away by the first product, so splitting it
% bounds the upload and nothing after it. The element axis bounds all four:
% an element's RF line is the quadrature sum over that element's own
% integration points, so elements are independent and the split is a pure
% re-blocking of the same arithmetic. The sensor axis is split as well, but
% only to bound the upload and the double cast the sparse product needs.
%
% Nathan Blanken, University of Twente, 2023

% Get number of transducer elements, number of integration points per
% element and number of dimensions:
[N_el,N_int,~] = size(Transducer.integration_points);
M = size(sensor_data.p,2); % Signal length
N_sensor = size(sensor_data.p,1);

% Get data casting properties:
[dataType, useGPU] = rf_data_cast(run_param);

apod    = Transducer.integration_receive_apodization;
delays  = Transducer.integration_receive_delays(:);
weights = Transducer.integration_weights;

% Normalise the quadrature weights:
if numel(weights) == 1
    weights = 1/N_int;
else
    weights = weights./sum(weights,2);
end

apod    = cast(apod,    dataType);
delays  = cast(delays,  dataType);
weights = cast(weights, dataType);

% Padded record length the delays need (shared with the preflight):
N = rf_signal_length(Transducer, Grid, M, dataType);

% Row r of sensor_weights is integration point r. define_sensor_transducer
% flattens integration_points column-major, so r = element + (i_int-1)*N_el
% and the element index varies fastest. delays(:) above and the reshape to
% [N_el N_int N] below both assume that ordering, and so does the row set of
% an element block. Check it rather than let a mismatch reshape silently.
if size(sensor_weights,1) ~= N_el*N_int
    error('PROTEUS:compute_RF:weightRows', ...
        ['sensor_weights has %d rows, expected N_el*N_int = %d. The ' ...
         'element blocking indexes its rows by integration point.'], ...
        size(sensor_weights,1), N_el*N_int);
end

% Set up frequency axis (Hz)
f = (0:(N-1))/(N*Grid.dt);
f = cast(f,dataType);

% Make symmetric around N/2 to keep time-domain signal real:
f(:,ceil(N/2+1):N) = -(f(:,floor(1+N/2):-1:2));

if useGPU
    f = gpuArray(f);
end

%% Partition both axes

[el_first, el_last] = rf_element_blocks(N_el, N_int, N, dataType, ...
    useGPU, run_param);
[s_first, s_last]   = rf_sensor_chunks(N_sensor, M, useGPU, run_param);
N_block = numel(el_first);
N_chunk = numel(s_first);

if N_block > 1 || N_chunk > 1
    run_log('banner', 'rfblocks', ...
        ['RF blocked into %d element block(s) x %d sensor chunk(s): ' ...
         '%d x %d and %d x %d against the %d-element gpuArray limit'], ...
        N_block, N_chunk, N_sensor, M, N_el*N_int, N, intmax('int32'));
end

% Hold the sensed pressure in its own precision, one chunk per device
% array, and cast to double only inside the product below: MATLAB has no
% single-precision sparse, so sensor_weights forces a double operand, and a
% double copy of the whole record is twice the array that did not fit.
sensor_p = cell(1, N_chunk);
if useGPU
    if N_chunk == 1
        % A no-op when run_simulation_homogeneous already left it on the
        % device; indexing with the full range would copy it instead.
        sensor_p{1} = gpuArray(sensor_data.p);
    else
        for c = 1:N_chunk
            sensor_p{c} = gpuArray(sensor_data.p(s_first(c):s_last(c),:));
        end
    end
else
    % Host arrays have no element limit, so the chunks exist only to bound
    % the double cast. Slicing per block would copy the record once per
    % block, so on one chunk keep the caller's array by reference.
    if N_chunk == 1
        sensor_p{1} = gather(sensor_data.p);
    else
        for c = 1:N_chunk
            sensor_p{c} = gather(sensor_data.p(s_first(c):s_last(c),:));
        end
    end
end

%% Per-element-block receive processing

p_all = zeros(N_el, N, dataType);

for b = 1:N_block

    el_block = el_first(b):el_last(b);
    n_el_b   = numel(el_block);

    % The block's rows, in the element-fastest order the reshape needs.
    rows_b = el_block(:) + (0:(N_int-1))*N_el;
    rows_b = rows_b(:);

    % Sensed pressure at this block's integration points. An element block
    % is a lateral band of the aperture, so most sensor chunks contribute
    % nothing to it and their sub-block is empty; a sparse product over an
    % empty block costs nothing, which is what keeps the total work equal
    % to the unblocked product.
    % Sliced by row once, not once per chunk: MATLAB stores a sparse matrix
    % by column, so taking a column range out of W_b below is nearly free
    % where re-indexing its rows walks the whole matrix again.
    W_b = sensor_weights(rows_b, :);

    p = [];
    for c = 1:N_chunk
        W_bc = W_b(:, s_first(c):s_last(c));
        if nnz(W_bc) == 0
            continue
        end
        if useGPU
            W_bc = gpuArray(W_bc);
        end
        contribution = W_bc*double(sensor_p{c});
        if isempty(p)
            p = contribution;
        else
            p = p + contribution;
        end
    end
    if isempty(p)
        % No sensor point carries this block: its elements record nothing.
        continue
    end
    p = cast(full(p),dataType);

    % Time shift in the frequency domain:
    delays_b = delays(rows_b);
    apod_b   = apod(el_block,:);
    if numel(weights) == 1
        weights_b = weights;
    else
        weights_b = weights(el_block,:);
    end
    if useGPU
        delays_b  = gpuArray(delays_b);
        apod_b    = gpuArray(apod_b);
        weights_b = gpuArray(weights_b);
    end

    p = fft(p,N,2);
    p = p.*exp(-2*pi*1i*delays_b*f);
    p = ifft(p,[],2,'symmetric');

    % Apply the receive apodization:
    p = reshape(p,n_el_b,N_int,N).* apod_b;

    % Compute the average pressure for each element:
    p = reshape(sum(p.*weights_b,2),n_el_b,N);

    if useGPU; p = gather(p); end

    p_all(el_block,:) = p;

end

% Convolution with receive impulse response. Outside the block loop: it acts
% along time on the assembled [N_el x N] result, which no grid takes over
% the limit, and convolving per block would pad every block separately.
IR = resample_signal(Transducer.ReceiveImpulseResponse, ...
    Transducer.SamplingRate, 1/Grid.dt, false);

RF = convn(p_all,IR)*Grid.dt;

end

