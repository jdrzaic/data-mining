#ifndef ERROR_H_
#define ERROR_H_


#include <stdexcept>
#include <string>


#define CHECK(status) \
    if (status != 0) \
        throw std::runtime_error( \
                std::string("Error in ") + \
                std::string(__FILE__) + \
                std::string(", line: ") + \
                std::to_string(__LINE__));


#endif  // ERROR_H_

