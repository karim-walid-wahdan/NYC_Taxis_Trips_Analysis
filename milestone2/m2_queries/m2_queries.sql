-- All trip info(location,tip amount,etc) .
SELECT 
    dataset.*,
    lookup_vendor.original_value AS vendor_decoded,
    lookup_flag.original_value AS store_and_fwd_flag_decoded,
    lookup_payment.original_value AS payment_type_decoded
FROM green_taxi_8_2019 dataset
INNER JOIN lookup_green_taxi_8_2019 lookup_vendor
    ON dataset.vendor = lookup_vendor.encoded_value
    AND lookup_vendor.column_name = 'vendor'
INNER JOIN lookup_green_taxi_8_2019 lookup_flag
    ON dataset.store_and_fwd_flag = lookup_flag.encoded_value
    AND lookup_flag.column_name = 'store_and_fwd_flag'
INNER JOIN lookup_green_taxi_8_2019 lookup_payment
    ON dataset.payment_type = lookup_payment.encoded_value
    AND lookup_payment.column_name = 'payment_type'
ORDER BY dataset.trip_distance DESC
LIMIT 20;

GO
-- What is the average fare amount per payment type.
SELECT AVG(dataset.fare_amount), lookup.original_value
FROM green_taxi_8_2019 dataset
INNER JOIN lookup_green_taxi_8_2019 lookup
     ON dataset.payment_type = lookup.encoded_value
     AND lookup.column_name = 'payment_type'
GROUP BY dataset.payment_type, lookup.original_value

GO 
-- On average, which city tips the most.
SELECT dataset.pu_area , AVG(dataset.tip_amount) AS average_tip_amount
FROM green_taxi_8_2019 dataset
GROUP BY dataset.pu_area
ORDER BY AVG(dataset.tip_amount) DESC 
LIMIT 1

GO
-- On average, which city tips the least.
SELECT dataset.pu_area , AVG(dataset.tip_amount) AS average_tip_amount
FROM green_taxi_8_2019 dataset
GROUP BY dataset.pu_area
ORDER BY AVG(dataset.tip_amount) 
LIMIT 1

GO 
-- What is the most frequent destination on the weekend.
SELECT dataset.do_area, COUNT(*) AS number_of_trips_weekend
FROM green_taxi_8_2019 dataset
WHERE dataset.is_weekend = 1
GROUP BY do_area
ORDER BY COUNT(*) DESC
LIMIT 1

GO 
-- On average, which trip type travels longer distances.
SELECT dataset.trip_type , AVG(dataset.trip_distance) AS average_trip_distance,lookup.original_value
FROM green_taxi_8_2019 dataset
INNER JOIN lookup_green_taxi_8_2019 lookup
    ON dataset.trip_type = lookup.encoded_value
    AND lookup.column_name = 'trip_type'
GROUP BY dataset.trip_type,lookup.original_value
ORDER BY AVG(dataset.trip_distance) DESC 
LIMIT 1

GO
-- between 4pm and 6pm what is the average fare amount.
SELECT AVG(dataset.fare_amount) AS average_fare_amount
FROM green_taxi_8_2019 dataset
WHERE dataset.lpep_dropoff_datetime_hour>=4 AND dataset.lpep_dropoff_datetime_hour<=6 OR
      dataset.lpep_pickup_datetime_hour>=4 AND dataset.lpep_pickup_datetime_hour<=6


