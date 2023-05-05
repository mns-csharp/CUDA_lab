//Example G.6 Header file cxtimers.h
//
// provides a MYTimer object for host bast elapsed time measurements.
// The timer depends on the C++ <chrono>
// usage: lap_ms() to returns interval since previous lap_ms(),
// start or reset.
#include <cstdio>
#include <cstdlib>
#include <chrono>

namespace cx 
{
    class timer 
	{
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> lap;
    public:
        timer()
		{ 
		    lap = std::chrono::high_resolution_clock::now(); 
		}
        void start() 
		{ 
		    lap = std::chrono::high_resolution_clock::now(); 
		}
        void reset() 
		{ 
		    lap = std::chrono::high_resolution_clock::now(); 
		}
        double lap_ms()
        {
            auto old_lap = lap;
            lap = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double,std::milli>
            time_span = (lap - old_lap);
            return (double)time_span.count();
        }
        double lap_us()
        {
            auto old_lap = lap;
            lap = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double,std::micro>
            time_span = (lap - old_lap);
            return (double)time_span.count();
        }
    };
}
