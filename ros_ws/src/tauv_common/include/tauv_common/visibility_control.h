#ifndef TAUV_COMMON__VISIBILITY_CONTROL_H_
#define TAUV_COMMON__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define TAUV_COMMON_EXPORT __attribute__ ((dllexport))
    #define TAUV_COMMON_IMPORT __attribute__ ((dllimport))
  #else
    #define TAUV_COMMON_EXPORT __declspec(dllexport)
    #define TAUV_COMMON_IMPORT __declspec(dllimport)
  #endif
  #ifdef TAUV_COMMON_BUILDING_LIBRARY
    #define TAUV_COMMON_PUBLIC TAUV_COMMON_EXPORT
  #else
    #define TAUV_COMMON_PUBLIC TAUV_COMMON_IMPORT
  #endif
  #define TAUV_COMMON_PUBLIC_TYPE TAUV_COMMON_PUBLIC
  #define TAUV_COMMON_LOCAL
#else
  #define TAUV_COMMON_EXPORT __attribute__ ((visibility("default")))
  #define TAUV_COMMON_IMPORT
  #if __GNUC__ >= 4
    #define TAUV_COMMON_PUBLIC __attribute__ ((visibility("default")))
    #define TAUV_COMMON_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define TAUV_COMMON_PUBLIC
    #define TAUV_COMMON_LOCAL
  #endif
  #define TAUV_COMMON_PUBLIC_TYPE
#endif

#endif  // TAUV_COMMON__VISIBILITY_CONTROL_H_
