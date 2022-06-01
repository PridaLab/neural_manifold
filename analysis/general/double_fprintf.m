function double_fprintf(fileID, varargin) %#ok<INUSL> 
    vars = varargin(2:end); %#ok<NASGU> 
    eval(strcat("fprintf('",varargin{1}, "',vars{:});"));
    eval(strcat("fprintf(fileID,'",varargin{1}, "',vars{:});"));
end

