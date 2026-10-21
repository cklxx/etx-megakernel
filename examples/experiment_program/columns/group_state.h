#pragma once
#include <stdint.h>
struct ColumnGroup { int owner,width,member[4],expert;float native_mix,mix[4]; };
struct ColumnGroupState {
 ColumnGroup group[24];unsigned long long up_done[24][37],down_done[24][37];unsigned long long epoch;int abort;
};
static_assert(sizeof(ColumnGroupState)<=16384,"graph group buffer bound");
