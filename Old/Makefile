make: gini split



split:
	g++ -shared -o cpp/libsplit.so -fPIC -O3 cpp/split.cpp

gini:
	g++ -shared -o cpp/libgini.so -fPIC -O3 cpp/gini.cpp

clean:
	rm cpp/libgini.so
