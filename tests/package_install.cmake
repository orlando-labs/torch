cmake_minimum_required(VERSION 3.27)
file(READ "${PACKAGES}/report.json" report)
string(JSON count LENGTH "${report}" packages)
math(EXPR last "${count}-1")
set(found FALSE)
foreach(index RANGE ${last})
  foreach(field native_adapter module_mode bundle)
    string(JSON ${field} GET "${report}" packages ${index} ${field})
  endforeach()
  if(native_adapter STREQUAL "prebuilt" AND module_mode STREQUAL "bytecode" AND bundle STREQUAL "runtime")
    foreach(field name version variant)
      string(JSON ${field} GET "${report}" packages ${index} ${field})
    endforeach()
    set(artifact "${PACKAGES}/${name}-${version}-${variant}.sputnikpkg")
    set(found TRUE)
    break()
  endif()
endforeach()
if(NOT found)
  message(FATAL_ERROR "Missing prebuilt bytecode/runtime package")
endif()
set(work "${PACKAGES}/verification/installed")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}/consumer")
file(COPY_FILE "${ROOT}/tests/package_smoke.s" "${work}/consumer/main.s")
set(environment "SPUTNIK_PKG_HOME=${work}/home"
  "SPUTNIK_NATIVE_PROVIDERS=" "SPUTNIK_NATIVE_EXT_CACHE=${work}/native-cache"
  "PYTORCH_ENABLE_MPS_FALLBACK=0")
set(cli "${SPUTNIK_ROOT}/build/sputnik")
execute_process(COMMAND cmake -E env ${environment}
  "${cli}" pkg install "${artifact}" --variant "${variant}" --bundle runtime --backend mps --allow-native
  WORKING_DIRECTORY "${work}/consumer" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND cmake -E env ${environment} SPUTNIK_NATIVE_CXX=/compiler-must-not-run
  "${cli}" run "${work}/consumer/main.s" --grant ffi
  WORKING_DIRECTORY "${work}/consumer" COMMAND_ERROR_IS_FATAL ANY)
file(WRITE "${work}/result.json"
  "{\"schema\":\"torch.package-install.v1\",\"variant\":\"${variant}\",\"install\":\"passed\",\"vm\":\"passed\",\"backends\":[\"cpu\",\"mps\"],\"mps_fallback\":false,\"external_providers\":false,\"runtime_compiler\":false}\n")
