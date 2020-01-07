cmake_minimum_required(VERSION 3.9.5)
 
project(se_ribs)
 
FIND_PACKAGE(ITK)

IF(ITK_FOUND)
    
  INCLUDE(${ITK_USE_FILE})

ELSE(ITK_FOUND)
    
  MESSAGE(FATAL_ERROR
    "ITK not found. Please set ITK_DIR.")

ENDIF(ITK_FOUND)
 
add_executable(se_ribs se_ribs.cxx)
target_link_libraries(se_ribs ${ITK_LIBRARIES})