#include <iostream>
#include "dmrg.h"
#include "parameters.h"
#include "ham_irlm.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    std::cout<<std::setprecision(15);
    time_t t0=time(nullptr);
    srand(t0);

    if (argc==4) {
        Parameters param;
        param.ReadParameters("parameters.txt");
        auto ham=HamIRLM(argv[1], argv[2], atof(argv[3]));
        t0=time(nullptr);
//        TestDMRGBasico(ham.Ham(), param);
//        CalculateNiN0();
    }
    else
        cout<<"usage: ./irlm <tfile> <Pfile> U\n";

    cout<<"\nDone in "<<difftime(time(nullptr),t0)<<"s"<<endl;
    return 0;
}
