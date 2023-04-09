#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "bitmap.hpp"

void test20()
{
   const rgb_t* colormap[4] = {
                                   hsv_colormap,
                                   jet_colormap,
                                 prism_colormap,
                                   vga_colormap
                              };

   const unsigned int fractal_width = 1200;
   const unsigned int fractal_height = 800;
   {
      bitmap_image fractal_hsv(fractal_width,fractal_height);
      bitmap_image fractal_jet(fractal_width,fractal_height);
      bitmap_image fractal_prism(fractal_width,fractal_height);
      bitmap_image fractal_vga(fractal_width,fractal_height);

      fractal_hsv.clear();
      fractal_jet.clear();
      fractal_prism.clear();
      fractal_vga.clear();

      double cr, ci;
      double nextr, nexti;
      double prevr, previ;

      const unsigned int max_iterations = 1000;

      for (unsigned int y = 0; y < fractal_height; ++y)
      {
         for (unsigned int x = 0; x < fractal_width; ++x)
         {
            cr = 1.5 * (2.0 * x / fractal_width  - 1.0) - 0.5;
            ci =       (2.0 * y / fractal_height - 1.0);

            nextr = nexti = 0;
            prevr = previ = 0;

            for (unsigned int i = 0; i < max_iterations; i++)
            {
               prevr = nextr;
               previ = nexti;

               nextr =     prevr * prevr - previ * previ + cr;
               nexti = 2 * prevr * previ + ci;

               if (((nextr * nextr) + (nexti * nexti)) > 4)
               {
                  if (max_iterations != i)
                  {
                     double z = sqrt(nextr * nextr + nexti * nexti);

                     #define log2(x) (std::log(1.0 * x) / std::log(2.0))

                     unsigned int index = static_cast<unsigned int>
                        (1000.0 * log2(1.75 + i - log2(log2(z))) / log2(max_iterations));
                     #undef log2

                     rgb_t c0 = colormap[0][index];
                     rgb_t c1 = colormap[1][index];
                     rgb_t c2 = colormap[2][index];
                     rgb_t c3 = colormap[3][index];

                     fractal_hsv.set_pixel(x, y, c0.red, c0.green, c0.blue);
                     fractal_jet.set_pixel(x, y, c1.red, c1.green, c1.blue);
                     fractal_prism.set_pixel(x, y, c2.red, c2.green, c2.blue);
                     fractal_vga.set_pixel(x, y, c3.red, c3.green, c3.blue);
                  }

                  break;
               }
            }
         }
      }

      fractal_hsv.save_image("test20_mandelbrot_set_hsv.bmp"  );
      fractal_jet.save_image("test20_mandelbrot_set_jet.bmp"  );
      fractal_prism.save_image("test20_mandelbrot_set_prism.bmp");
      fractal_vga.save_image("test20_mandelbrot_set_vga.bmp"  );
   }
}

int main()
{
   test20();
   return 0;
}
