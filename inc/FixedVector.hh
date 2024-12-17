#ifndef FIXED_VECTOR_H
#define FIXED_VECTOR_H

#include <stdexcept> // out_of_range() and for future use.
#include <stdio.h>
#include <vector>

template <typename T>
class FixedVector
{
private:
  std::vector<T> vec;

public:
  FixedVector() : vec() {}
  explicit FixedVector(size_t size, T default_value = T()) : vec(size, default_value) {}

  T& operator[](std::size_t idx)
  {
    if (idx >= vec.size())
      throw std::out_of_range("Index out of bounds. Vec size: " + std::to_string(vec.size()));
<<<<<<< HEAD
    return vec[idx];
=======
    return vec[vec.size() - idx - 1];
>>>>>>> 80422cb (Initial UT folder)
  }

  const T& operator[](size_t idx) const
  {
    if (idx >= vec.size())
      throw std::out_of_range("Index out of bounds");
    return vec[idx];
  }

  void push(FixedVector<T> new_val)
  {
    /*
      Places new value at the front, moves all other elements
      one position to the right.
    */
    if (vec.size() == 0)
      throw std::runtime_error("Cannot push on an empty FixedVector");
    vec.erase(vec.begin());
    vec.push_back(new_val);
  }

  void push(T input){
    if(vec.size() == 0)
      throw std::runtime_error("Cannot push element into vector of size 0!");
    vec.erase(vec.begin());
    vec.push_back(input);
  }

  auto size() { return vec.size(); }
  auto size() const { return this->vec.size(); }

  auto begin() { return vec.begin(); }       // Mutable
  auto end() { return vec.end(); }
  auto begin() const { return vec.begin(); } // Const
  auto end() const { return vec.end(); }

  size_t size() const {
    return vec.size();
  }

  typename std::vector<T>::iterator begin() {
    return vec.begin();
  }

  typename std::vector<T>::iterator end() {
    return vec.end();
  }

  typename std::vector<T>::const_iterator begin() const {
    return vec.begin();
  }

  typename std::vector<T>::const_iterator end() const {
    return vec.end();
  }

  // Disable operations that change size
  void push_back(const int&) = delete;
  void emplace_back(int) = delete;
  void resize(size_t) = delete;
};

#endif