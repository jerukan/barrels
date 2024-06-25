function xyzs = randCylPointsSurf(model, npoints, sigma)
    % generates uniformly distributed points on the surface of a cylinder
    axis = model.Orientation;
    axis = axis / norm(axis);
    axnull = null(axis);
    axnull1 = axnull(:, 1);
    axnull2 = axnull(:, 2);
    x1 = model.Parameters(1:3);
    r = model.Radius;
    h = model.Height;
    % top and bottom caps
    cap_area = 2 * pi * r^2;
    side_area = h * pi * r * 2;
    % for a single cap
    ncap = fix((cap_area / (cap_area + side_area)) * npoints / 2);
    nside = npoints - 2 * ncap;

    rand_theta = rand(1, npoints) * 2 * pi;
    rand_h = rand(1, npoints) * h;
    rand_r = zeros(1, npoints) + r;
    rand_r(1:2*ncap) = r * sqrt(rand(1, 2 * ncap));
    rand_r(2*ncap+1:end) = rand_r(2*ncap+1:end) + normrnd(0, sigma, 1, nside);
    rand_h(1:ncap) = 0 + normrnd(0, sigma, ncap, 1);
    rand_h(ncap+1:2*ncap) = h + normrnd(0, sigma, 1, ncap);
    cosval = repmat(axnull1, 1, npoints) .* rand_r .* cos(rand_theta);
    sinval = repmat(axnull2, 1, npoints) .* rand_r .* sin(rand_theta);
    xyzs = cosval + sinval + repmat(x1', 1, npoints) + rand_h .* repmat(axis', 1, npoints);

    xyzs = xyzs';
end
