#pragma once

#ifdef PKAI_HOST
    #include <pkai/host/connection/fully_connected.hpp>
#else
    #ifdef PKAI_DEVICE
        #include <pkai/device/connection/fully_connected.hpp>
    #else
        #error PKAI_HOST and PKAI_DEVICE both undefined
    #endif
#endif