set -e                              
BUILD=build                        
rm -rf "$BUILD"
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$BUILD"
"./$BUILD/olegrad" "$@"
