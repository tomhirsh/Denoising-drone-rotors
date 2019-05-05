Fs = 1/(T(2)-T(1));    
prevIdx = 0;
rpm(1) = 0;
for i = 2:1:size(Enc3,1)
    rpm(i) = rpm(i-1);
    if Enc3(i) > 0
        tmpRpm = 60*Fs/(i-prevIdx);
        if (tmpRpm < 10000)
            prevIdx = i;
            rpm(i) = tmpRpm;
        end
    end
end
