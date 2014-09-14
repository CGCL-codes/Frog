#ifndef TIMER_GUARD
#define TIMER_GUARD

#if defined(__linux) || defined(__unix__) || defined(unix) || defined (__unix)
  // L!nux/Un!x: use gettimeofday() function
#  define TIMER_USE_GETTIMEOFDAY
#  include <sys/time.h>
#elif defined(_WIN32)
  // Windows: use QueryPerformance*() functions
#  define TIMER_USE_QUERY_PERFORMANCE
  // do not include <windows.h> to avoid namespace pollution;
  // define the necessary prototypes by hand instead.
  extern "C" int __stdcall QueryPerformanceCounter(long long int*);
  extern "C" int __stdcall QueryPerformanceFrequency(long long int*);
#elif (__cplusplus >= 201103L)
  // C++11 compiler; use <chrono> header clock
  // which is not yet supported by all compilers
#  define TIMER_USE_STD_CHRONO
#  include <chrono>
#else
  // Fallback implementation: use ANSI-C clock() function
  // which counts cycles, so if things run in parallel, the measured time
  // goes up and must be divided by the number of processes
#  include <ctime>
#endif

/**
 * Implements a small stopwatch-like timer
 */
class Timer
{
  private:
#if defined(TIMER_USE_GETTIMEOFDAY)
    timeval _start;
    timeval _end;
    static void stamp(timeval& point)
    {
      gettimeofday(&point, NULL);
    }
    double elapsed() const
    {
      return double(_end.tv_sec-_start.tv_sec) + 1E-6 * double(_end.tv_usec - _start.tv_usec);
    }
#elif defined(TIMER_USE_QUERY_PERFORMANCE)
    long long int _start;
    long long int _end;
    static void stamp(long long int& point)
    {
      QueryPerformanceCounter(&point);
    }
    double elapsed() const
    {
      long long int freq = 0ll;
      QueryPerformanceFrequency(&freq);
      return (freq == 0ll) ? 0.0 : double(_end - _start) / double(freq);
    }
#elif defined(TIMER_USE_STD_CHRONO)
    typedef std::chrono::high_resolution_clock Clock;
    Clock::time_point _start;
    Clock::time_point _end;
    static void stamp(Clock::time_point& point)
    {
      point = Clock::now();
    }
    double elapsed() const
    {
      return 1E-6 * double(std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count());
    }
#else
    clock_t _start;
    clock_t _end;
    static void stamp(clock_t& point)
    {
      point = ::clock();
    }
    double elapsed() const
    {
      return double(_end - _start) / double(CLOCKS_PER_SEC);
    }
#endif

  public:
    // start the stopwatch
    void start()
    {
      stamp(_start);
    }

    // stop time measurement and return elapsed time in seconds
    double stop()
    {
      stamp(_end);
      return elapsed();
    }

};


#endif  // define TIMER_GUARD
