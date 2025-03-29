// Load Sri Lanka boundaries
var sri_lanka = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq('ADM0_NAME', 'Sri Lanka'));
Map.centerObject(sri_lanka, 7);

// SMAP L4 (9 km daily)
var smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/004")
  .filterDate("2020-01-01", "2024-12-31")
  .select("soil_moisture")
  .filterBounds(sri_lanka);

// Clip to Sri Lanka and calculate monthly mean
var smap_monthly = smap.map(function(img) {
  return img.clip(sri_lanka);
}).mean();

// Sentinel-1 GRD (10 m resolution, but resampled to 500 m for SM)
var sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(sri_lanka)
  .filterDate("2023-01-01", "2023-12-31")
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV');

  // MODIS LST (1 km)
var modis_lst = ee.ImageCollection("MODIS/061/MOD11A1")
.filterBounds(sri_lanka)
.select('LST_Day_1km');

// Export SMAP (9 km)
Export.image.toDrive({
    image: smap_monthly,
    description: "SMAP_SriLanka_2020_2023",
    scale: 9000,
    region: sri_lanka.geometry(),
    fileFormat: "GeoTIFF"
  });
  
  // Export Sentinel-1 (500 m)
  Export.image.toDrive({
    image: sentinel1.mean(),
    description: "Sentinel1_SriLanka_VV_2020_2023",
    scale: 500,
    region: sri_lanka.geometry(),
    fileFormat: "GeoTIFF"
  });