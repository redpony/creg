#ifndef _RULE_LEXER_H_
#define _RULE_LEXER_H_

#include <iostream>
#include <string>
#include <utility>

struct JSONFeatureMapLexer {
  typedef void (*FeatureMapCallback)(const std::string& id,
                                     const std::pair<int,float>* begin,
                                     const std::pair<int,float>* end,
                                     void* extra);
  static void ReadRules(std::istream* in, FeatureMapCallback func, void* extra);
};

#endif

