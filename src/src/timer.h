#ifndef TIMER_GUARD
#define TIMER_GUARD

#if defined(__linux) || defined(__unix__) || defined(unix) || defined (__unix)
// Linux/Unix: use gettimeofday() function
#define TIMER_USE_GETTIMEOFDAY
#include <sys/time.h>
#elif defined(_WIN32)
// Windows: use QueryPerformance*() functions
// Should be INCLUDED AFTER <windows.h> to avoid conflicts if <windows.h> is included
#define TIMER_USE_QUERY_PERFORMANCE
	#ifndef _WINDOWS_
	// avoid conflicts with <windows.h>
	// define the necessary prototypes by hand instead.
		#ifdef __cplusplus
			extern "C" int __stdcall QueryPerformanceCounter(long long int*);
			extern "C" int __stdcall QueryPerformanceFrequency(long long int*);
		#else
			extern int __stdcall QueryPerformanceCounter(long long int*);
			extern int __stdcall QueryPerformanceFrequency(long long int*);
		#endif
	#endif
#elif (__cplusplus >= 201103L)
// C++11 compiler; use <chrono> header clock
// which is not yet supported by all compilers
#define TIMER_USE_STD_CHRONO
#include <chrono>
#else
// Fallback implementation: use ANSI-C clock() function
// which counts cycles, so if things run in parallel, the measured time
// goes up and must be divided by the number of processes
#include <ctime>
#endif

//returns time interval in ms

#if defined(TIMER_USE_GETTIMEOFDAY)
static struct timeval _start;
static struct timeval _end;
static void stamp(struct timeval * point)
{
    gettimeofday(point, NULL);
}
static double elapsed()
{
    return 1000.0 * (_end.tv_sec-_start.tv_sec) + 0.001 * (_end.tv_usec - _start.tv_usec);
}
#elif defined(TIMER_USE_QUERY_PERFORMANCE)
static long long int _start;
static long long int _end;
static long long int _freq = 0ll;
static double _freq_interval = 0.0;
static void stamp(long long int * point)
{
#ifndef _WINDOWS_
	QueryPerformanceCounter(point);
#else
	// avoid conflicts with <windows.h>
	QueryPerformanceCounter((LARGE_INTEGER *)point);
#endif
}
static double elapsed()
{
    if (_freq == 0ll)
    {
	#ifndef _WINDOWS_
        QueryPerformanceFrequency(&_freq);
	#else
		// avoid conflicts with <windows.h>
        QueryPerformanceFrequency((LARGE_INTEGER *)&_freq);
	#endif
        if (_freq != 0ll) _freq_interval = 1000.0 / _freq;
    }
    return (_freq == 0ll) ? 0.0 : _freq_interval * (_end - _start);
}
#elif defined(TIMER_USE_STD_CHRONO)
typedef std::chrono::high_resolution_clock Clock;
static Clock::time_point _start;
static Clock::time_point _end;
static void stamp(Clock::time_point * point)
{
    *point = Clock::now();
}
static double elapsed()
{
    return 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
}
#else
static clock_t _start;
static clock_t _end;
static void stamp(clock_t * point)
{
    *point = clock();
}
static double elapsed()
{
    return (_end - _start) * 1000.0 / CLOCKS_PER_SEC;
}
#endif

// start the stopwatch
static void timer_start()
{
    stamp(&_start);
}

// stop time measurement and return elapsed time in ms
static double timer_stop()
{
    stamp(&_end);
    return elapsed();
}

#endif  // define TIMER_GUARD
