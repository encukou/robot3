BODY_THICKNESS = 2;
BODY_R = 65;
RIM_W = 3;
RIM_H = 2;

WHEEL_OUT = 39;
WHEEL_GAP_R = 55;
WHEEL_SPACE_R = 60/2;
WHEEL_TH = 17;

CENTER_HOLE_R = 15/2;

MOTOR_H = 22.4;
MOTOR_W = 18;
MOTOR_OUT = 16;
HOLDER_WALL_W = 4;
MOTOR_SHAFT_POS = 10.92;
MOTOR_SHAFT_R = 3.49;
MOTOR_PIN_POS = 22.35;
MOTOR_PIN_R = 3.68/2;
MOTOR_PIN_L = 2;
MOTOR_SCREW_POS = 31.17;
MOTOR_SCREW_R = 2.64/2;
MOTOR_SCREW_SEP = 17.42;
HOLDER_LEN = 42;
NUT_R = 3;
NUT_H = 2;

MOTOR_FLAP_W = 3;
MOTOR_FLAP_S = 7;
MOTOR_FLAP_L = 6;
MOTOR_FLAP_HOLE_POS = 2.54;
MOTOR_FLAP_HOLE_R = 1/2;

TOL = 0.25;

EPS = 0.01;

module box (sizes, centering=[1, 1, 1]) {
    translate ([
        -sizes[0]*centering[0]/2,
        -sizes[1]*centering[1]/2,
        -sizes[2]*centering[2]/2]) cube (sizes);
}

module sym3 () {
    for (r=[0, 1, 2]) rotate ([0, 0, 360/3*r]) {
        children();
    }
}

module base_plate () {
    difference () {
        union () {
            cylinder (BODY_THICKNESS+RIM_H, r=BODY_R, $fn=100);
        }
        translate ([0, 0, BODY_THICKNESS]) {
            cylinder (RIM_H+EPS, r1=BODY_R-RIM_W*2, r2=BODY_R-RIM_W, $fn=100);
        }
        translate ([0, 0, -EPS]) {
            cylinder (BODY_THICKNESS+EPS*2, r1=CENTER_HOLE_R+BODY_THICKNESS, r2=CENTER_HOLE_R, $fn=100);
        }
        sym3() {
            translate ([WHEEL_OUT, 0, BODY_THICKNESS+MOTOR_H/2]) rotate ([0, 90, 0]) {
                cylinder (100, r=WHEEL_SPACE_R);
            }
        }
    }
}

module drop_cylinder(h, r) {
    intersection () {
        hull () for (x=[0, 1]) translate ([-x*r*1.5, 0, 0]) {
            cylinder (h, r=r*(1-x)+TOL);
        }
        box ([(r+TOL)*2, (r+TOL)*2, h+1000]);
    }
}

module motor_holder_third () {
    translate ([MOTOR_OUT, -MOTOR_SHAFT_POS, BODY_THICKNESS]) {
        difference () {
            union () {
                // Inside (main) wall
                translate ([-HOLDER_WALL_W, -HOLDER_WALL_W, 0]) cube ([HOLDER_WALL_W, HOLDER_LEN, MOTOR_H]);
                // Back of motor
                // XXX: These aren't calculated parametrically
                _MOTOR_OVERHANG = 17;
                R = HOLDER_WALL_W/2+0.23;
                WALL_W = MOTOR_FLAP_L-TOL;
                LOWER_OVERHANG = 5;
                translate ([-_MOTOR_OVERHANG, -WALL_W, 0]) {
                    cube ([_MOTOR_OVERHANG+WALL_W/2+MOTOR_W/2, WALL_W, MOTOR_H]);
                    translate ([-LOWER_OVERHANG, WALL_W/2, 0]) {
                        cube ([_MOTOR_OVERHANG, WALL_W/2, MOTOR_H/2-MOTOR_SHAFT_R-TOL]);
                    }
                }
                // Connecting to another holder
                translate ([-_MOTOR_OVERHANG, -R, 0]) cylinder (MOTOR_H, r=R);
                L = HOLDER_WALL_W*1.7;
                translate ([-HOLDER_WALL_W*2+EPS, MOTOR_PIN_POS-L/2, 0]) cube ([HOLDER_WALL_W, L, MOTOR_H]) ;
            }
            // Shaft/pin holes
            for (pr=[
                [MOTOR_PIN_POS, MOTOR_PIN_R, MOTOR_PIN_L],
                [MOTOR_SHAFT_POS, MOTOR_SHAFT_R, HOLDER_WALL_W],
            ]) {
                pos = pr[0];
                r = pr[1];
                xpos = pr[2];
                hull () for (z=[0, 1]) translate ([-xpos-TOL, pos, MOTOR_H/2+z*100]) {
                    rotate ([0, 90, 0]) cylinder (100, r=r);
                }
            }
            // Screw holes
            for (pr=[
                [MOTOR_SCREW_POS, MOTOR_SCREW_R, MOTOR_SCREW_SEP/2],
                [MOTOR_SCREW_POS, MOTOR_SCREW_R, -MOTOR_SCREW_SEP/2],
            ]) {
                pos = pr[0];
                r = pr[1];
                zpos = pr[2];
                translate ([0, pos, MOTOR_H/2+zpos]) union () {
                    rotate ([0, 90, 0]) translate ([0, 0, -50]) drop_cylinder (100, r);
                    rotate ([0, 90, 0]) translate ([0, 0, -HOLDER_WALL_W+TOL+NUT_H-100]) {
                        rotate ([0, 0, 90]) cylinder (100, r=NUT_R+TOL/2, $fn=6);
                    }
                }
            }
            // Motor flap
            translate ([MOTOR_W/2-TOL, -MOTOR_FLAP_L/2, MOTOR_H/2]) {
                translate ([0, 0, -MOTOR_FLAP_S/2]) box ([MOTOR_FLAP_W+TOL, MOTOR_FLAP_L*100, 100], [0, 1, 0]);
                // screw hole??
                *rotate ([0, 90, 0]) translate ([0, 0, -40]) drop_cylinder (40+EPS, r=1);
            }
        }
        // Flap nub
        translate ([MOTOR_FLAP_HOLE_R-TOL, -MOTOR_FLAP_HOLE_POS, MOTOR_H/2]) {
            intersection () {
                rotate ([0, 90, 0]) cylinder (MOTOR_W/2, r=MOTOR_FLAP_HOLE_R);
                translate ([(MOTOR_W)/2, 0, 0]) intersection () {
                    rotate ([0, -45, 0]) box ([2, 2, 2], [2, 1, 0]);
                    box ([2, MOTOR_FLAP_HOLE_R*2-TOL, 2], [1, 1, 1]);
                }
            }
        }
    }
}

module motor_holder () {
    sym3 () motor_holder_third ($fn=50);
    translate ([0, 0, BODY_THICKNESS]) difference () {
        cylinder (MOTOR_H/2-MOTOR_SHAFT_R-TOL, r=MOTOR_OUT);
        translate ([0, 0, -EPS]) cylinder (100, r=MOTOR_OUT-HOLDER_WALL_W);
    }
}

%base_plate ();
motor_holder ();

// Helpers

module motor_assembly() {
    translate ([0, 0, -19]) {
        translate ([9.25, -25.5, 6.25]) rotate ([0, -90, 0]) color ([1, 1, 0]) {
            import("Geared_Motor.stl", convexity=3);
        }
        scale ([1, 1, -1]) translate ([0, 0, 1]) color ([0, 1, 1]) {
            import ("wheel-adapter.stl", convexity=3);
        }
        translate ([0, 0, -6]) scale ([1, 1, -1]) color ([0, 0, 1]) {
            cylinder (20, r=53/2);
        }
    }
}

// Motors
*%sym3 () rotate([0, 0, 90]) {
    translate ([0, -16, 13]) rotate ([90+180, 90, 0]) motor_assembly();
}

// Tile size ref
%translate ([0, 0, -2]) %difference () {
    cylinder (1, r=100);
    translate ([0, 0, -1]) cylinder (3, r=65);
}

// Octopus board
%translate ([7, 0, 0]) cube ([65, 65, 1], center=true);

// Turtle pen
*%cylinder (20, d=15);

// Screws
%sym3 () translate ([MOTOR_OUT, -MOTOR_SHAFT_POS, BODY_THICKNESS]) for (i=[-1, 1]) {
    color ([0, 0, 0]) translate ([19, MOTOR_SCREW_POS, MOTOR_H/2+i*MOTOR_SCREW_SEP/2]) {
        rotate ([0, -90, 0]) cylinder (25, d=3, $fn=50);
        rotate ([0, -90, 0]) translate ([0, 0, -1]) cylinder (1, d=5, $fn=6);
    }
}

