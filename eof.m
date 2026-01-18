function eof()
    syms LT; % length from center of wheel to center of mass
    syms R; % wheel radius
    syms b; % wheel spacing
    syms mB; % body mass
    syms IB; % body inertia
    syms mW; % wheel mass
    syms IW; % wheel inertia
    syms IV; % axis & wheel inertia (along "z" axis)
    syms g; % gravity
    
    syms theta dtheta ddtheta; % segway tilt
    syms phi dphi ddphi; % segway compass orientation
    syms x dx ddx; % segway forward speed

    syms TL TR; % wheel torques
    syms Tphi Tx; % EOF forces
    Tphi = (TL-TR)*R/b;
    Tx = (TL+TR)*R;


    % Potential energy
    Ep = LT * cos(theta) * g * mB;


    syms dxA dyA; % Axis center movement
    syms dxT dyT; % COM movement
    syms dxL dyL; % Left wheel movement
    syms dxR dyR; % Right wheel movement
    syms dxsiL dxsiR; % Wheels rotational movement
    dxA = dx * cos(phi);
    dyA = dx * sin(phi);
    dxT = dxA + LT*(dtheta*cos(theta)*cos(phi)-dphi*sin(theta)*sin(phi));
    dyT = dyA + LT*(dtheta*cos(theta)*sin(phi)-dphi*sin(theta)*cos(phi));
    dzT = -LT * dtheta*sin(theta);
    dxL = dxA + 1/2*b*dphi*cos(phi);
    dyL = dyA + 1/2*b*dphi*sin(phi);
    dxR = dxA - 1/2*b*dphi*cos(phi);
    dyR = dyA - 1/2*b*dphi*sin(phi);
    dxsiR = dx/R + (b*dphi)/(2*R);
    dxsiL = dx/R - (b*dphi)/(2*R);

    
    % Kinetic energy
    Ek = 1/2*mB*(dxT^2+dyT^2+dzT^2) ... % segway linear movement kinetic energy
        + 1/2*dtheta^2*IB ... % segway tilting kinetic energy
        + 1/2*mW*(dxR^2+dyR^2+dxL^2+dyL^2) ... % wheels kinetic energy
        + 1/2*IW*(dxsiR^2 + dxsiL^2) ... % wheels kinetic rotation energy
        + 1/2*dphi^2*IV; % segway rotation kinetic energy
    
    

    L = expand(Ek - Ep);
   
    q = [theta, phi, x];
    dq = [dtheta, dphi, dx]; 
    ddq = [ddtheta, ddphi, ddx]; 
    M = [0, Tphi, Tx];

    EL_eqns = sym(zeros(size(q)));
    for i = 1:length(q)
        dLdq = diff(L, q(i));
        dLdqdot = diff(L, dq(i));

        ddt_dLdqdot = 0;
        for j = 1:length(q)
            ddt_dLdqdot = ddt_dLdqdot + sum(diff(dLdqdot, q(j)) * dq(j));
            ddt_dLdqdot = ddt_dLdqdot + sum(diff(dLdqdot, dq(j)) * ddq(j));
        end
        EL_eqns(i) = simplify(ddt_dLdqdot - dLdq - M(i));
    end
    [eq2, eq4, eq6] = solve(EL_eqns, ddtheta, ddphi, ddx);
       
    python_format(simplify(eq2))
    python_format(simplify(eq4))
    python_format(simplify(eq6))

    JA = jacobian([eq2, eq4, eq6], [theta, dtheta, phi, dphi, x, dx]);
    JB = jacobian([eq2, eq4, eq6], [TL TR]);
    eqs_at_eqA = subs(JA, [theta, dtheta, phi, dphi, x, dx, TL TR],  {0, 0, 0, 0, 0, 0, 0, 0});
    eqs_at_eqB = subs(JB, [theta, dtheta, phi, dphi, x, dx, TL TR],  {0, 0, 0, 0, 0, 0, 0, 0});

    A = [0,1,0,0,0,0; eqs_at_eqA(1,:); 0,0,0,1,0,0; eqs_at_eqA(2,:); 0,0,0,0,0,1; eqs_at_eqA(3,:)];
    A = subs(A, [IB, IW, IV, R, b, LT, mB, mW, g],  {0.1, 0.02, 0.05, 0.05, 0.3, 0.2, 1, 0.5, 9.81})
    B = [0,0; eqs_at_eqB(1,:); 0,0; eqs_at_eqB(2,:); 0,0; eqs_at_eqB(3,:)];
    B = subs(B, [IB, IW, IV, R, b, LT, mB, mW, g],  {0.1, 0.02, 0.05, 0.05, 0.3, 0.2, 1, 0.5, 9.81})
    A = double(A);
    B = double(B);
    Q = [1, 0, 0, 0, 0, 0;
        0, 1, 0, 0, 0, 0;
        0, 0, 1, 0, 0, 0;
        0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 1000000, 0;
        0, 0, 0, 0, 0, 1;
        ];
    R = eye(size(B,2));
    [K, S, E] = lqr(A, B, Q, R)
end



function python_format(expr)
    py_expr = char(expr);
    py_expr = strrep(py_expr, 'sin', 'np.sin');
    py_expr = strrep(py_expr, 'cos', 'np.cos');
    py_expr = strrep(py_expr, '^', '**');
    py_expr = strrep(py_expr, '*', ' * ');
    py_expr = strrep(py_expr, '/', ' / ');
    py_expr = strrep(py_expr, '-', ' - ');
    py_expr = strrep(py_expr, '+', ' + ');
    py_expr = strrep(py_expr, ' ', '');
    py_expr = ['(', py_expr, ')'];
    fprintf('%s\n\r', py_expr);
end


