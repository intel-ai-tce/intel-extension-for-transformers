//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <chrono>
namespace dispatcher_utils {
using namespace std;
using namespace std::chrono;
class Timer {
 public:
  void start() { m_start = high_resolution_clock::now(); }
  void stop() { m_end = high_resolution_clock::now(); }
  long long get_elapsed_time() const { return duration_cast<microseconds>(m_end - m_start).count(); }

 private:
  high_resolution_clock::time_point m_start;
  high_resolution_clock::time_point m_end;
};
}  // namespace dispatcher_utils