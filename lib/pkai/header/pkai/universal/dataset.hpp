#pragma once

#ifdef PKAI_HOST
#include <pkai/host/dataset.hpp>
#else
#ifdef PKAI_DEVICE
#include <pkai/device/dataset.hpp>
#else
#error PKAI_HOST and PKAI_DEVICE both undefined
#endif
#endif