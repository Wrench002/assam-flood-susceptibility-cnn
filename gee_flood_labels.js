var districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.and(
    ee.Filter.eq('ADM1_NAME', 'Assam'),
    ee.Filter.inList('ADM2_NAME',
      ['Barpeta','Dhemaji','Nagaon','Morigaon','Lakhimpur'])
  ));

var AOI = districts.geometry();

Map.centerObject(AOI, 8);

var labels = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence')
  .gt(20)
  .clip(AOI)
  .rename('flood');

Map.addLayer(labels, {min:0, max:1, palette:['black','cyan']}, 'Flood Labels');

Export.image.toDrive({
  image: labels,
  description: 'flood_labels_assam',
  folder: 'DHARA_Assam',
  fileNamePrefix: 'flood_labels_assam',
  scale: 30,
  region: AOI,
  maxPixels: 1e11
});
