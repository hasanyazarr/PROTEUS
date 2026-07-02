function hash = file_hash(filename)
%FILE_HASH Return a SHA-256 hash for a file, or '' when unavailable.

hash = '';
if isempty(filename) || ~exist(filename, 'file')
    return
end

try
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(filename, 'r');
    if fid < 0
        return
    end
    cleaner = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    hash = lower(reshape(dec2hex(digest)', 1, []));
catch
    hash = '';
end

end
