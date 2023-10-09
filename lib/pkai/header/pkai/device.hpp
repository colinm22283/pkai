#pragma once

#ifdef PKAI_HOST
#error Both PKAI_DEVICE and PKAI_HOST were defined
#endif

#define PKAI_DEVICE

constexpr bool PKAI_HOST_CONSTEXPR = false;
constexpr bool PKAI_DEVICE_CONSTEXPR = true;