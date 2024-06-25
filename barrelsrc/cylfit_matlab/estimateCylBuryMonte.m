function burialVolume = estimateCylBuryMonte(model, npoints)
    % monte carlo estimation of cylinder burial
    xyzs = randCylPoints(model, npoints);
    undermask = xyzs(: ,3) <= 0;
    cylvol = cylVolume(model);
    burialVolume = cylvol * (sum(undermask) / npoints);
end
