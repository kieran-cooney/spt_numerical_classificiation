abelian_group_cocyle:= function(a,b,c)
    return (a[2] - b[2])*(c[1]-b[1]);
end;

abelian_right_boundary_operator_phase:= function(gl, gr, h, p, v)
    return (v(gl, h+p, p) + v(gl-h, gr, p)) - v(gl, gr, p);
end;

abelian_right_linear_rep_phase:= function(gl, gr, f, h, p, v)
    return (abelian_right_boundary_operator_phase(gl, gr, f+h, p, v) - abelian_right_boundary_operator_phase(gl, gr, f, p, v) - abelian_right_boundary_operator_phase(gl, gr, h, p, v));
end;

right_boundary_operator_phase:= function(gl, gr, h, p, v)
    return (v(gl, h*p, p)*v((h^-1)*gl, gr, p))/v(gl, gr, p);
end;

right_linear_rep_phase:= function(gl, gr, f, h, p, v)
    return (right_boundary_operator_phase(gl, gr, f*h, p, v)/(right_boundary_operator_phase(gl, gr, f, p, v)*right_boundary_operator_phase(gl, gr, h, p, v)));
end;

test := function(x);
    return x*x;
end;
