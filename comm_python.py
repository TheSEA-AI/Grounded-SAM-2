import os
import shutil

# Function to get file names without extensions
def get_file_names_without_extension(directory):
    # List all files in the directory (excluding directories)
    return {os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))}

# Directories to compare
hed_dir = "./hed_4-1"
rmd_dir = "./removed_4-1"
dst_dir = "./removed_hed"

# Get file names without extensions
files_in_hed = get_file_names_without_extension(hed_dir)
files_in_rmd = get_file_names_without_extension(rmd_dir)

# Find common file names (intersection of the two sets)
common_files = files_in_hed.intersection(files_in_rmd)
print(len(common_files))

# Print common file names without extensions
print("Common files (without extensions):")
for file in common_files:
    #print(file)
    shutil.move(os.path.join(hed_dir, file+'.png'), os.path.join(dst_dir, file+'.png'))


# infer_dataset 88639
# hed 77855
# removed 10785
# outline | mask 153047
'''
ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/hed_sam2auto_seg/ | wc -l
77855

ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/removed | wc -l
10785

ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/contour_sam2auto_seg/ | wc -l 
ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/contour_hed_sam2auto_seg/ | wc -l
ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/mask_hed_sam2auto_seg/ | wc -l


ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/contour_sam2auto_seg/ | wc -l
153047
ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/contour_hed_sam2au
to_seg/ | wc -l
153047
ls -l /home/yangmi/s3data/outline_mask_hed/max_1e-4/mask_hed_sam2auto_
seg/ | wc -l
115407
ls -l ./mask_hed_sam2auto_seg/ | wc  -l
153047

ls -l ./removed_6/ | wc  -l
10794
(SAM2) [yangmi@ip-172-31-7-210 Grounded-SAM-2]$ ls -l ./mask_hed_6/ | wc  -l
152555
(SAM2) [yangmi@ip-172-31-7-210 Grounded-SAM-2]$ ls -l ./contour_6/ | wc  -l
152555
(SAM2) [yangmi@ip-172-31-7-210 Grounded-SAM-2]$ ls -l ./contour_hed_6/ | wc  -l
152555
ls -l ./hed_6/ | wc  -l
77846


(SAM2) [yangmi@ip-172-31-7-210 Grounded-SAM-2]$ ls -l ./hed_sam2auto_seg/ | wc  -l
88314
(SAM2) [yangmi@ip-172-31-7-210 Grounded-SAM-2]$ ls -l ./removed/ | wc  -l
10780
(SAM2) [yangmi@ip-172-31-7-210 Grounded-SAM-2]$ ls -l ./mask_hed_sam2auto_seg/ | wc  -l
152545
'''