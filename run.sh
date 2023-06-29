#!/bin/bash

case "$1" in
	f)
		python src/filter_song.py
		;;
	fft)
		python src/test_fft.py
		;;
	i)
		python src/instruments.py
		;;
	b)
		python src/basics.py
		;;
	r)
		shift
		python src/main.py "$@"
		;;
	*)
		echo "unknown option"
		;;
esac
