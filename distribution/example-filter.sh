#!/bin/bash
curl -k "https://localhost:8080/api/filter" \
	--data "George Washington was president and his ssn was 123-45-6789 and he lived in 90210." -H "Content-type: text/plain"
