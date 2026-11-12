function syncdata = synchronize(file,method,Fs)

disp(strjoin(["Extracting data from",file,"..."]));
if(strcmp(method,"zero_order_hold"))
    syncdata=sync_zoh_exa(file,Fs);
elseif(strcmp(method,"resampling"))
    syncdata=sync_resample_exa(file,Fs);
else
    error("Undefined resampling method. The method must be either ""zero_order_hold"" or ""resampling"".");
end

disp(strjoin(["Data from",file,"loaded and synchronized successfully."]));

end