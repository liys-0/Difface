import csv
import os

def main():
    dataset_dir = "Difface/faceclip/dataset"
    meta_path = os.path.join(dataset_dir, "mock_meta296_full.csv")
    snp_path = os.path.join(dataset_dir, "mock_snp_ATGC_with_LOG10P.csv")
    out_path = os.path.join(dataset_dir, "mock_snp_processed_800.csv")

    # 1. Read meta file and rank geno_id
    meta_records = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['img_id']
            geno_id = row['geno_id']
            img_num = int(''.join(filter(str.isdigit, img_id)))
            meta_records.append((img_num, img_id, geno_id))
    
    meta_records.sort(key=lambda x: x[0])
    ranked_img_ids = [r[1] for r in meta_records]
    ranked_geno_ids = [r[2] for r in meta_records]

    # Genotype string to integer mapping
    GENO_MAP = {
        'AA': 1, 'TT': 2, 'CC': 3, 'GG': 4,
        'AT': 5, 'TA': 5,
        'AC': 6, 'CA': 6,
        'AG': 7, 'GA': 7,
        'TC': 8, 'CT': 8,
        'TG': 9, 'GT': 9,
        'CG': 10, 'GC': 10
    }

    # 2. Read SNP data
    threshold = 1.42
    selected_snps = []
    
    with open(snp_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        subject_cols = header[5:]
        
        if len(subject_cols) != len(ranked_img_ids):
            print(f"Warning: {len(subject_cols)} subjects in SNP file but {len(ranked_img_ids)} in meta file.")

        for row in reader:
            if not row:
                continue
            log10p = float(row[4])
            if log10p >= threshold:
                chrom = row[0]
                pos = row[1]
                ref = row[2]
                alt = row[3]
                chrom_pos_name = f"{chrom}:{pos} {ref}>{alt}"
                
                # Map each genotype string to its corresponding integer
                vals = [GENO_MAP.get(val, -1) for val in row[5:]]
                
                selected_snps.append((chrom_pos_name, vals))
                
    print(f"Selected {len(selected_snps)} SNPs with LOG10P >= {threshold}")

    # 3 & 4. Generate new CSV
    out_header = ['image_id', 'geno_id'] + [snp[0] for snp in selected_snps]
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        
        for i in range(len(ranked_img_ids)):
            row_out = [ranked_img_ids[i], ranked_geno_ids[i]]
            for snp in selected_snps:
                row_out.append(snp[1][i])
            writer.writerow(row_out)
            
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    main()
