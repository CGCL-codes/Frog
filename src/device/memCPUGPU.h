#pragma once
#include "stdio.h"
#include "stdlib.h"

#define __STREAM__

#ifdef __STREAM__
#include "../device/memStream.h"
#else
#include "./device/memDefault.h"
#endif
