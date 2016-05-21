#include <fstream>

#include "../include/model.h"

using namespace std;
using namespace gcl;

int main()
{
    ifstream fobj("cube.obj");

    auto m = wavefront_loader(fobj);

    for(auto x : m.attr_normal) {
        cout << x << endl;
    }
}

