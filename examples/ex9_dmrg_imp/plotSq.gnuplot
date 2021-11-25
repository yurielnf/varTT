set term postscript enhanced color

set key autotitle columnhead
set key top right

set output 'renyi.ps'


set multiplot layout 2,1 title "TB chain for L=16 OBC, m=128"

set title "chain geometry"
plot for[i=2:7] "entropy.dat" u 1:i w lp

set title "star geometry"
plot for[i=2:7] "entropy_star.dat" u 1:i w lp

unset multiplot
