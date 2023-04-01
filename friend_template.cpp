#include <iostream>

template<typename t>
class Manager
{
private:
    t * x;
    int length;

public:
    Manager()
    {
        x = nullptr;
        length = 0;
    }
    void set(int i, t val)
    {
        x[i]=val;
    }
    t get(int i)
    {
        return x[i];
    }
    void allocate(int len)
    {
		length = len;
        x = new t[length];
    }
    int get_length()
    {
        return length;
    }
    //template<typename t2> friend void init_data_func(Manager& manager);
    void display()
    {
        for(int i=0 ; i<length ; i++)
        {
            std::cout<<x[i]<<", ";
        }
    }
    void init_data(void (*my_func)(Manager &))
    {
        my_func(*this);
    }
};

void init_data_(Manager<float> &my_obj)
{
    for (int i = 0; i < my_obj.get_length(); ++i)
    {
        my_obj.set(i, rand());
    }
}

int main()
{
    const int length = 10;
    Manager<float> my_obj;
    my_obj.allocate(length);
    my_obj.init_data(init_data_);
    my_obj.display();
}
