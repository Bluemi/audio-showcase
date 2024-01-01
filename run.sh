#!/bin/bash

case "$1" in
	f)
		shift
		python src/filter_song.py "$@"
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
	s)
		shift
		python src/scalogram.py "$@"
		;;
	is)
		shift
		python src/interactive-spectogram.py "$@"
		;;
	*)
		echo "unknown option"
		;;
esac
