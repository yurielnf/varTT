#include <iostream>
#include <tensor.h>

using namespace std;

int main() {
  cout << "hola mundo!" << endl;

  TensorD t({1, 2, 3});
  t.FillRandu();
  t.Save(cout);
  return 0;
}
