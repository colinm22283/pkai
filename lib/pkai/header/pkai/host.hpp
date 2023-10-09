#pragma once

#ifdef PKAI_DEVICE
#error Both PKAI_DEVICE and PKAI_HOST were defined
#endif

#define PKAI_HOST

#include <pkai/universal/network_builder.hpp>
#include <pkai/universal/layer.hpp>