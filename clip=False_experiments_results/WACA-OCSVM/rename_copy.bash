for file in clip=*-*.txt; do
    if [[ $file =~ clip=([^-]+)-(.*)_Mean_EER_(.*)_df_test_dict.txt ]]; then
        scaler_clip=$(echo $file | cut -d'-' -f1 | cut -d'=' -f2)
        smoothing=$(echo $file | cut -d'-' -f2- | cut -d'_' -f1)
        preprocessing_method=$(echo $file | rev | cut -d'_' -f4- | rev)
	

	# Rename the file
    	echo "----------------------------"
	echo "original file name: $file"
    	echo "new file name: $new_filename"
    	echo "----------------------------"
        new_filename="clip=${scaler_clip}-Smoothing=${smoothing}-Prep=${preprocessing_method}-EER_df_test_dict.txt"
        cp "$file" "$new_filename"
    else
        echo "Skipping file $file because it does not match the expected format."
    fi
done
