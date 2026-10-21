// Generated from export_contract.json by export_contract.cpp().
#pragma once
static size_t row_intermediate_values(const std::string& role,const std::vector<char>& args) {
  if(role=="row_dsl_moe_up") { int value; memcpy(&value,args.data()+56,4); if(value<0)exit(1); return size_t(value)*2; }
  if(role=="row_dsl_moe_down") { return 0; }
  if(role=="row_dsl_gemv_fold") { int value; memcpy(&value,args.data()+52,4); if(value<0)exit(1); return size_t(value)*1; }
  fprintf(stderr,"unknown row export role\n");exit(1);
}
