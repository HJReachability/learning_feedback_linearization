
syms x y z psi theta phi dx  dy  dz zeta xi p  q  r Ix Iy Iz m



g_17 = -1/m*(sin(phi)*sin(psi) +cos(phi)*cos(psi)*sin(theta));
g_18 = -1/m*(cos(psi)*sin(phi)- cos(phi)*sin(psi)*sin(theta));
g_19 = -1/m*cos(phi)*cos(theta);

f =        [dx;
            dy;
            dz;
            q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta);
            q*cos(phi) - r*sin(phi);
            p + q*(sin(phi)*tan(theta)) + r*(cos(phi)*tan(theta));
            g_17*zeta;
            g_18*zeta;
            g_19*zeta;
            xi;
            0;
            (Iy-Iz)/Ix*q*r;
            (Iz -Ix)/Iy*p*r;
            (Ix -Iy)/Iz *p*q];






h_1 = x;
h_2 = y;
h_3 = z;
h_4 = psi;

g_1 = [0 ;0 ;0 ;0 ;0 ;0 ; 0; 0; 0 ;0 ; 1 ; 0 ;  0; 0 ];
g_2 = [0 ;0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0; 0 ; 0 ; 0 ; 1/Ix ; 0 ; 0];
g_3 = [0 ;0 ;0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1/Iy ; 0];
g_4 = [0 ; 0 ;0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1/Iz];

%first output Lee derivatives
L_1fh_1 = simplify(jacobian(h_1,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * f);

L_2fh_1 = simplify(jacobian(L_1fh_1,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] )*f);
L_3fh_1 = simplify(jacobian(L_2fh_1, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * f);
L_4fh_1 = simplify(jacobian(L_3fh_1, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * f);

Lg_1L_3fh_1 = simplify(jacobian(L_3fh_1,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_1);
Lg_2L_3fh_1 = simplify(jacobian(L_3fh_1, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * g_2);
Lg_3L_3fh_1 = simplify(jacobian(L_3fh_1, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * g_3);
Lg_4L_3fh_1 = simplify(jacobian(L_3fh_1, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * g_4);

%Second output Lee derivatives


L_1fh_2 = simplify(jacobian(h_2, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * f);
L_2fh_2 = simplify(jacobian(L_1fh_2,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] )*f);
L_3fh_2 = simplify(jacobian(L_2fh_2,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * f);
L_4fh_2 = simplify(jacobian(L_3fh_2,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * f);

Lg_1L_3fh_2 = simplify(jacobian(L_3fh_2,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * g_1);
Lg_2L_3fh_2 = simplify(jacobian(L_3fh_2, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * g_2);
Lg_3L_3fh_2 = simplify(jacobian(L_3fh_2,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_3);
Lg_4L_3fh_2 = simplify(jacobian(L_3fh_2, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_4);


%Third output Lee derivatives

L_1fh_3 = simplify(jacobian(h_3, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * f);
L_2fh_3 = simplify(jacobian(L_1fh_3,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ])*f);
L_3fh_3 = simplify(jacobian(L_2fh_3,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r ]) * f);
L_4fh_3 = simplify(jacobian(L_3fh_3,[ x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * f);


Lg_1L_3fh_3 = simplify(jacobian(L_3fh_3, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_1);
Lg_2L_3fh_3 = simplify(jacobian(L_3fh_3, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_2);
Lg_3L_3fh_3 = simplify(jacobian(L_3fh_3, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_3);
Lg_4L_3fh_3 = simplify(jacobian(L_3fh_3, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_4);

%Third output Lee derivatives

L_1fh_4 = simplify(jacobian(h_4, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * f);
L_2fh_4 = simplify(jacobian(L_1fh_4, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] )*f);

Lg_1L_1fh_4 = simplify(jacobian(L_1fh_4, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_1);
Lg_2L_1fh_4 = simplify(jacobian(L_1fh_4, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_2);
Lg_3L_1fh_4 = simplify(jacobian(L_1fh_4, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_3);
Lg_4L_1fh_4 = simplify(jacobian(L_1fh_4, [x, y, z, psi, theta, phi, dx , dy , dz, zeta, xi, p , q , r] ) * g_4);



 L_4fh_1
 L_4fh_2
 L_4fh_3
 L_2fh_4

[Lg_1L_3fh_1, Lg_2L_3fh_1, Lg_3L_3fh_1, Lg_4L_3fh_1;
 Lg_1L_3fh_2, Lg_2L_3fh_2, Lg_3L_3fh_2, Lg_4L_3fh_2;
 Lg_1L_3fh_3, Lg_2L_3fh_3, Lg_3L_3fh_3, Lg_4L_3fh_3;
 Lg_1L_1fh_4, Lg_2L_1fh_4, Lg_3L_1fh_4, Lg_4L_1fh_4
]

[h_1 ; L_1fh_1; L_2fh_1; L_3fh_1;h_2; L_1fh_2; L_2fh_2; L_3fh_2; h_3; L_1fh_3; L_2fh_3; L_3fh_3; h_4; L_1fh_4]
