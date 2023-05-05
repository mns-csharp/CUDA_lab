# include "pngwriter.hpp"

int main()
{
	
}

//# include "bitmap.hpp"

/*
void makePicturePNG(int *Mandel,int width, int height, int MAX)
{
    double red_value, green_value, blue_value;
    float scale = 256.0/MAX;
    double MyPalette[41][3] = 
	{
        {1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0},// 0, 1, 2, 3, 
        {1.0,1.0,1.0},{1.0,0.7,1.0},{1.0,0.7,1.0},{1.0,0.7,1.0},// 4, 5, 6, 7,
        {0.97,0.5,0.94},{0.97,0.5,0.94},{0.94,0.25,0.88},{0.94,0.25,0.88},//8, 9, 10, 11,
        {0.91,0.12,0.81},{0.88,0.06,0.75},{0.85,0.03,0.69},{0.82,0.015,0.63},//12, 13, 14, 15, 
        {0.78,0.008,0.56},{0.75,0.004,0.50},{0.72,0.0,0.44},{0.69,0.0,0.37},//16, 17, 18, 19,
        {0.66,0.0,0.31},{0.63,0.0,0.25},{0.60,0.0,0.19},{0.56,0.0,0.13},//20, 21, 22, 23,
        {0.53,0.0,0.06},{0.5,0.0,0.0},{0.47,0.06,0.0},{0.44,0.12,0},//24, 25, 26, 27, 
        {0.41,0.18,0.0},{0.38,0.25,0.0},{0.35,0.31,0.0},{0.31,0.38,0.0},//28, 29, 30, 31,
        {0.28,0.44,0.0},{0.25,0.50,0.0},{0.22,0.56,0.0},{0.19,0.63,0.0},//32, 33, 34, 35,
        {0.16,0.69,0.0},{0.13,0.75,0.0},{0.06,0.88,0.0},{0.03,0.94,0.0},//36, 37, 38, 39,
        {0.0,0.0,0.0}//40
    };

    int i;
    int iy;
    pngwriter png(width,height,1.0,"Mandelbrot.png");   
    for (int j=height-1; j>=0; j--) 
	{
        for (int i=0; i<width; i++) 
		{
            // compute index to the palette
            int indx= (int) floor(5.0*scale*log2f(1.0f*Mandel[j*width+i]+1));
            red_value=MyPalette[indx][0];
            green_value=MyPalette[indx][2];
            blue_value=MyPalette[indx][1];
            png.plot(i,j, red_value, green_value, blue_value);            
        }
    }
    png.close();
}
*/

/*
void MandelbrotSet(int WIDTH, int HEIGHT, int max_iterations)
{
   const rgb_t * colormap = prism_colormap;

   const unsigned int fractal_width = 1200;
   const unsigned int fractal_height = 800;
   {
      bitmap_image fractal_prism(fractal_width,fractal_height);

      fractal_prism.clear();

      double cr, ci;
      double nextr, nexti;
      double prevr, previ;

      //const unsigned int max_iterations = 1000;

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

                     rgb_t c2 = colormap[index];

                     fractal_prism.set_pixel(x, y, c2.red, c2.green, c2.blue);
                  }

                  break;
               }
            }
         }
      }

      fractal_prism.save_image("mandelbrot_set_prism.bmp");
   }
}


int main(int argc, char **argv) 
{
    //Set pixel size {H,V} 
    int WIDTH = 1200; 
	int HEIGHT = 800;
    
	//Determine number of sampling iterations {ITER}
    int ITER = 1000;

    //Allocate an array to hold the result
    int *Iters = (int*) malloc(sizeof(int)*WIDTH*HEIGHT);

	MandelbrotSet(WIDTH, HEIGHT, ITER);
        
	//makePicturePNG(Iters, WIDTH, HEIGHT, ITER);
}
*/