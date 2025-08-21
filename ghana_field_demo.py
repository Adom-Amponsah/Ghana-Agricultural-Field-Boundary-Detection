import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage
import pandas as pd
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import cv2

# Configure page
st.set_page_config(
    page_title="Ghana Agricultural Field Boundary Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Ghana theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #006B3C 0%, #FCD116 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #006B3C;
    }
    .stButton > button {
        background-color: #006B3C;
        color: white;
        border: none;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# YOUR WORKING PRODUCTION-BASED FIELD DELINEATION CLASS
# ============================================================================

class ProductionBasedFieldDelineation:
    """
    Field delineation specifically for production/yield raster data
    This works with your maize production values, not raw satellite imagery
    """

    def __init__(self, production_data):
        self.data = production_data
        self.height, self.width = production_data.shape

    def segment_fields_by_production(self, method='watershed'):
        """Segment fields based on production values"""

        if method == 'watershed':
            return self._watershed_production_segmentation()
        elif method == 'kmeans':
            return self._kmeans_production_segmentation()
        elif method == 'threshold':
            return self._threshold_based_segmentation()
        elif method == 'contour':
            return self._contour_based_segmentation()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _watershed_production_segmentation(self):
        """Use watershed on production data"""
        # Normalize data for processing
        normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)

        # Apply Gaussian blur to smooth the data
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

        # Find local maxima using scipy.ndimage.maximum_filter as an alternative
        neighborhood_size = 3
        local_max = (blurred == ndimage.maximum_filter(blurred, size=neighborhood_size))
        # Further thresholding to get distinct peaks
        peaks = blurred * local_max
        coordinates = np.argwhere(peaks > (0.3 * peaks.max()))

        # Create markers from coordinates
        markers = np.zeros_like(blurred, dtype=int)
        for i, coord in enumerate(coordinates):
            markers[coord[0], coord[1]] = i + 1

        # If no peaks found, create some markers
        if len(coordinates) == 0:
            markers[blurred == blurred.max()] = 1

        # Apply watershed
        labels = segmentation.watershed(-blurred, markers, mask=self.data > 0)

        return labels

    def _kmeans_production_segmentation(self, n_clusters=5):
        """Use K-means clustering on production values"""
        # Get valid (non-zero) production values
        valid_mask = self.data > 0
        valid_data = self.data[valid_mask]

        if len(valid_data) == 0:
            return np.zeros_like(self.data)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(valid_data.reshape(-1, 1))

        # Map back to full image
        result = np.zeros_like(self.data, dtype=int)
        result[valid_mask] = cluster_labels + 1  # +1 to avoid confusion with background (0)

        return result

    def _threshold_based_segmentation(self):
        """Simple threshold-based segmentation"""
        # Create production categories
        valid_data = self.data[self.data > 0]
        if len(valid_data) == 0:
            return np.zeros_like(self.data)

        high_prod = self.data > np.percentile(valid_data, 75)
        med_prod = (self.data > np.percentile(valid_data, 25)) & \
                   (self.data <= np.percentile(valid_data, 75))
        low_prod = (self.data > 0) & (self.data <= np.percentile(valid_data, 25))

        result = np.zeros_like(self.data, dtype=int)
        result[low_prod] = 1
        result[med_prod] = 2
        result[high_prod] = 3

        return result

    def _contour_based_segmentation(self):
        """Use contours to find field boundaries"""
        # Normalize data
        normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create labeled image with proper data type
        result = np.zeros_like(self.data, dtype=np.int32)  # Use int32 instead of int

        for i, contour in enumerate(contours):
            cv2.drawContours(result, [contour], -1, i + 1, thickness=cv2.FILLED)

        return result

# ============================================================================
# YOUR WORKING VISUALIZATION FUNCTION
# ============================================================================

def visualize_field_delineation(original_data, segmented_data, method_name):
    """Visualize the field delineation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original production data
    im1 = axes[0, 0].imshow(original_data, cmap='YlOrRd')
    axes[0, 0].set_title('Original Maize Production Data')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])

    # Segmented fields
    im2 = axes[0, 1].imshow(segmented_data, cmap='tab20')
    axes[0, 1].set_title(f'Field Segments - {method_name}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])

    # Field boundaries overlay
    boundaries = segmentation.find_boundaries(segmented_data, mode='thick')
    overlay = original_data.copy()
    overlay[boundaries] = overlay.max()  # Highlight boundaries

    im3 = axes[1, 0].imshow(overlay, cmap='YlOrRd')
    axes[1, 0].set_title('Production + Field Boundaries')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])

    # Statistics
    unique_labels = np.unique(segmented_data)
    # Exclude background label (0) if it exists
    field_labels = unique_labels[unique_labels != 0]
    n_fields = len(field_labels)

    avg_field_size = np.count_nonzero(segmented_data) / n_fields if n_fields > 0 else 0
    total_production_area = np.count_nonzero(original_data)

    axes[1, 1].text(0.1, 0.8, f'Number of fields: {n_fields}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'Average field size: {avg_field_size:.1f} pixels', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Total production area: {total_production_area} pixels', fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'Max production: {original_data.max():.2f}', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Field Statistics')

    plt.tight_layout()
    return fig

def load_and_analyze_tiff(uploaded_file):
    """Load TIFF and analyze the actual data type"""
    bytes_data = uploaded_file.read()
    with rasterio.MemoryFile(bytes_data) as memfile:
        with memfile.open() as src:
            data = src.read(1)  # Read first (and likely only) band
            meta = src.meta

            st.write(f"Data shape: {data.shape}")
            st.write(f"Data type: {data.dtype}")
            st.write(f"Data range: {data.min()} to {data.max()}")
            st.write(f"Non-zero values: {np.count_nonzero(data)}")

            return data, meta

def calculate_field_stats(boundaries_mask, pixel_size_m2=100):
    """
    Calculate field statistics from boundary mask
    """
    # Label connected components (fields)
    labeled_fields = measure.label(~boundaries_mask, connectivity=2)
    
    # Get properties
    properties = measure.regionprops(labeled_fields)
    
    # Calculate statistics
    field_areas = [prop.area * pixel_size_m2 / 10000 for prop in properties]  # Convert to hectares
    field_count = len(properties)
    total_area = sum(field_areas)
    avg_area = np.mean(field_areas) if field_areas else 0
    
    return {
        'field_count': field_count,
        'total_area_ha': total_area,
        'average_field_size_ha': avg_area,
        'field_areas': field_areas,
        'labeled_fields': labeled_fields
    }

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Ghana Agricultural Field Boundary Detection</h1>
        <p>AI-Powered Field Delineation for Sustainable Agriculture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🛠️ Processing Settings")
    
    # Detection method selection - using YOUR working methods
    method = st.sidebar.selectbox(
        "Detection Method",
        ["watershed", "kmeans", "threshold", "contour"],
        help="Choose the algorithm for field boundary detection"
    )
    
    # K-means clusters (only for kmeans method)
    if method == 'kmeans':
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of production zones for K-means clustering"
        )
    
    # Pixel size
    pixel_size = st.sidebar.number_input(
        "Pixel Size (m²)",
        min_value=1,
        max_value=1000,
        value=100,
        help="Area represented by each pixel"
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Satellite Image",
        type=['tif', 'tiff'],
        help="Upload .tif file from AGWAA or satellite imagery"
    )
    
    # Demo data option
    use_demo = st.sidebar.button("🎯 Use Demo Data")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Process uploaded file or demo
    if uploaded_file is not None or use_demo:
        
        if use_demo:
            # Create synthetic demo data similar to your Ghana maize production data
            np.random.seed(42)
            demo_image = np.zeros((77, 53))  # Same dimensions as your actual data
            
            # Add some field-like patterns with production values
            for i in range(8):
                y, x = np.random.randint(5, 70), np.random.randint(5, 48)
                size_y, size_x = np.random.randint(8, 15), np.random.randint(8, 12)
                production_value = np.random.uniform(2.0, 8.5)  # Realistic maize production values
                demo_image[y:y+size_y, x:x+size_x] = production_value
            
            production_data = demo_image
            metadata = None
            st.sidebar.success("✅ Demo data loaded!")
            
        else:
            # Load uploaded file using YOUR function
            try:
                production_data, metadata = load_and_analyze_tiff(uploaded_file)
                st.sidebar.success(f"✅ {uploaded_file.name} loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        # Process using YOUR working pipeline
        with st.spinner("🔄 Processing field boundaries using your production-based methods..."):
            
            # Initialize YOUR field delineation class
            delineator = ProductionBasedFieldDelineation(production_data)
            
            # Run YOUR segmentation method
            try:
                if method == 'kmeans':
                    # Use the n_clusters parameter for kmeans
                    original_method = delineator._kmeans_production_segmentation
                    delineator._kmeans_production_segmentation = lambda: original_method(n_clusters)
                
                segmented = delineator.segment_fields_by_production(method)
                
                # Calculate boundaries for statistics
                boundaries = segmentation.find_boundaries(segmented, mode='thick')
                field_stats = calculate_field_stats(boundaries, pixel_size)
                
                st.success(f"✅ Field delineation complete using {method}!")
                
            except Exception as e:
                st.error(f"❌ Error with {method}: {e}")
                return
        
        # Display results using YOUR visualization
        st.subheader("📊 Field Delineation Results")
        
        # Use YOUR working visualization function
        fig = visualize_field_delineation(production_data, segmented, method.title())
        st.pyplot(fig)
        
        # Statistics using the same format as your original code
        st.subheader("📊 Field Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🌾 Fields Detected",
                f"{field_stats['field_count']:,}"
            )
        
        with col2:
            st.metric(
                "📏 Total Area",
                f"{field_stats['total_area_ha']:.2f} ha"
            )
        
        with col3:
            st.metric(
                "📐 Average Field Size",
                f"{field_stats['average_field_size_ha']:.2f} ha"
            )
        
        with col4:
            st.metric(
                "🎯 Processing Method",
                method.title()
            )
        
        # Field size distribution
        if field_stats['field_areas']:
            st.subheader("📈 Field Size Distribution")
            
            # Create histogram
            fig = px.histogram(
                x=field_stats['field_areas'],
                nbins=20,
                title="Distribution of Field Sizes",
                labels={'x': 'Field Size (hectares)', 'y': 'Number of Fields'},
                color_discrete_sequence=['#006B3C']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Boundary Map"):
                # Create downloadable image
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(production_data, cmap='YlOrRd', alpha=0.7)
                boundaries = segmentation.find_boundaries(segmented, mode='thick')
                ax.imshow(boundaries, cmap='Reds', alpha=0.8)
                ax.set_title("Ghana Field Boundaries")
                ax.axis('off')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Download PNG",
                    data=buf.getvalue(),
                    file_name="ghana_field_boundaries.png",
                    mime="image/png"
                )
        
        with col2:
            if st.button("📊 Download Statistics"):
                # Create CSV with field data
                df = pd.DataFrame({
                    'Field_ID': range(1, len(field_stats['field_areas']) + 1),
                    'Area_Hectares': field_stats['field_areas']
                })
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="ghana_field_statistics.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Generate Report"):
                # Generate summary report
                report = f"""
Ghana Field Boundary Analysis Report
===================================

Processing Details:
- Method: {method.title()}
- Pixel Size: {pixel_size} m²

Results Summary:
- Fields Detected: {field_stats['field_count']}
- Total Area: {field_stats['total_area_ha']:.2f} hectares
- Average Field Size: {field_stats['average_field_size_ha']:.2f} hectares
"""
                if field_stats['field_areas']:
                    report += f"""- Smallest Field: {min(field_stats['field_areas']):.2f} hectares
- Largest Field: {max(field_stats['field_areas']):.2f} hectares"""
                
                report += f"""

Generated using AI-powered field boundary detection
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="ghana_field_report.txt",
                    mime="text/plain"
                )
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to Ghana Field Boundary Detection
        
        This AI-powered tool helps detect and analyze agricultural field boundaries from production data.
         
        ### 📋 How to use:
        1. **Upload** your .tif production data file (like Ghana_Predicted_Maize_Production_2024.tif)
        2. **Choose** from your proven methods: watershed, kmeans, threshold, or contour
        3. **View** detected field boundaries and statistics
        4. **Download** results for further analysis
        
        ### 🎯 Try it now:
        - Click "Use Demo Data" in the sidebar for a quick demonstration
        - Or upload your own TIFF production file
        
        ### 🌍 Built for Ghana:
        - Uses your proven production-based field delineation methods
        - Optimized for small production raster files
        - Ready for integration into mobile apps/dashboards
        """)

if __name__ == "__main__":
    main()