#include "f_util.h"

void setStr(std::vector<char>& array, std::string string) {
  for (int i = 0; i < string.size(); i++) {
    array[i] = string[i];
  }
  array[string.size()] = 0;
}

bool matchStr(const std::vector<char>& array, std::string string) {
  bool allmatch = true;
  for (int i = 0; i < string.size(); i++) {
    allmatch = array[i] == string[i];
  }
  return allmatch;
}
