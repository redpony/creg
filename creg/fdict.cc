#include "fdict.h"
#include <string>
#include <sstream>

using namespace std;

Dict FD::dict_;
bool FD::frozen_ = false;

std::string FD::Convert(std::vector<WordID> const& v) {
    return Convert(&*v.begin(),&*v.end());
}

std::string FD::Convert(WordID const *b,WordID const* e) {
  ostringstream o;
  for (WordID const* i=b;i<e;++i) {
    if (i>b) o << ' ';
    o << FD::Convert(*i);
  }
  return o.str();
}

