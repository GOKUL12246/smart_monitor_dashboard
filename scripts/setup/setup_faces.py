#!/usr/bin/env python3
"""
📸 CREATE SAMPLE MISSING PERSONS DATABASE
Quick setup with sample face data
"""

import os
import requests
from pathlib import Path
import time

def download_sample_faces():
    """Download sample faces from Lorem Picsum"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING SAMPLE FACE IMAGES")
    print("="*60)
    
    # Use Lorem Picsum which provides random faces
    # Format: https://picsum.photos/400/400?random=N
    
    sample_persons = [
        {'name': 'Sample_Person_1', 'count': 2},
        {'name': 'Sample_Person_2', 'count': 2},
        {'name': 'Sample_Person_3', 'count': 2},
    ]
    
    missing_dir = Path('missing_persons')
    missing_dir.mkdir(exist_ok=True)
    
    for person_idx, person_info in enumerate(sample_persons):
        person_name = person_info['name']
        person_dir = missing_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        print(f"\n👤 {person_name}")
        
        for img_idx in range(person_info['count']):
            filename = f"{person_name}_{img_idx+1}.jpg"
            filepath = person_dir / filename
            
            try:
                # Use picsum.photos or another source
                url = f"https://picsum.photos/400/400?random={person_idx}_{img_idx}" 
                
                print(f"   Downloading: {filename}...", end=" ", flush=True)
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print("✅")
                else:
                    print(f"❌ (Status {response.status_code})")
            
            except Exception as e:
                print(f"❌ ({str(e)[:30]})")
            
            time.sleep(0.5)  # Rate limiting
    
    print("\n✅ Sample download complete!")


def create_synthetic_faces():
    """Create synthetic test faces using OpenCV"""
    print("\n" + "="*60)
    print("🎨 CREATING SYNTHETIC TEST FACES")
    print("="*60)
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    sample_persons = [
        {'name': 'Test_Person_1', 'color': (100, 150, 200)},
        {'name': 'Test_Person_2', 'color': (200, 100, 150)},
        {'name': 'Test_Person_3', 'color': (150, 200, 100)},
    ]
    
    missing_dir = Path('missing_persons')
    missing_dir.mkdir(exist_ok=True)
    
    for person in sample_persons:
        person_name = person['name']
        person_dir = missing_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        print(f"\n👤 {person_name}")
        
        # Create multiple variations
        for variant in range(3):
            filename = f"{person_name}_{variant+1}.jpg"
            filepath = person_dir / filename
            
            # Create a test image with a colored square (face placeholder)
            img = np.ones((400, 400, 3), dtype=np.uint8) * 50
            
            # Draw a circle (face)
            center = (200, 200)
            radius = 80 + variant * 10
            color = person['color']
            cv2.circle(img, center, radius, color, -1)
            
            # Add some features
            cv2.circle(img, (180, 180), 10, (255, 255, 255), -1)
            cv2.circle(img, (220, 180), 10, (255, 255, 255), -1)
            cv2.line(img, (190, 220), (210, 220), (0, 0, 0), 3)
            
            # Add person name
            cv2.putText(img, person_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(str(filepath), img)
            print(f"   ✅ Created: {filename}")
    
    print("\n✅ Synthetic faces created!")
    return True


def show_status():
    """Show current status"""
    print("\n" + "="*60)
    print("📊 DATABASE STATUS")
    print("="*60)
    
    missing_dir = Path('missing_persons')
    
    if not missing_dir.exists():
        print("No database yet")
        return
    
    total_images = 0
    total_people = 0
    
    for person_dir in sorted(missing_dir.iterdir()):
        if person_dir.is_dir():
            images = list(person_dir.glob('*.[jp][pn]g'))
            if images:
                total_people += 1
                total_images += len(images)
                print(f"✅ {person_dir.name:25} - {len(images)} image(s)")
    
    if total_people == 0:
        print("❌ No images found")
    else:
        print(f"\n📊 Total: {total_people} people, {total_images} images")


def main():
    print("\n" + "="*60)
    print("📸 SAMPLE FACE DATABASE SETUP")
    print("="*60)
    
    print("\nOptions:")
    print("1. Create synthetic test faces (fast, no internet needed)")
    print("2. Download real faces from internet (slow, needs internet)")
    print("3. Show current status")
    print("4. Exit")
    
    choice = input("\nChoose (1-4): ").strip()
    
    if choice == '1':
        create_synthetic_faces()
        show_status()
        print("\n✨ Ready! Run: python temple_monitor.py")
    
    elif choice == '2':
        try:
            download_sample_faces()
            show_status()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try option 1 instead (no internet needed)")
    
    elif choice == '3':
        show_status()
    
    else:
        print("Bye!")


if __name__ == '__main__':
    main()
