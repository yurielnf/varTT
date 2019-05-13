 g++ -O3 -std=c++11 ../*.cpp ../../../src/*.cpp -pg \
     -I ../../../src \
     -larmadillo -llapacke

##export LD_LIBRARY_PATH="../../../src/:$LD_LIBRARY_PATH"
##     -L/home/yurielnf/Documents/projects/build-vartt_suite-Desktop_Qt_5_9_1_GCC_64bit-Release/src -lvartt \

./a.out $1; 
gprof a.out gmon.out  > analysis.txt
