# Contents

 - `data` - Data used in the project
 - `env` - Files required for setting up the environment
 - `src` - Source files used throughout the code
 - `conf.py` - Configuration file for the project

# Comptetition description

Taken from the original site

## ABOUT

Booking.com’s mission is to make it easier for everyone to experience the world. By investing in the technology that
helps take the friction out of travel, Booking.com seamlessly connects millions of travellers with memorable
experiences, a range of transport options and incredible places to stay. 
Many of the travellers go on trips which
include more than one destination. For instance, a user from the US could fly to Amsterdam for 5 nights, then spend 2
nights in Brussels, 3 in Paris and 1 in Amsterdam again before heading back home. In this scenario, we suggest options
for extending their trip immediately they make their booking.
The goal of this challenge is to use a dataset based on millions of real anonymized accommodation reservations to come
up with a strategy for making the best recommendation for their next destination in real-time.

## Dataset

The challenge is part of the WebTour 2021 ACM WSDM workshop on web tourism that will be held at the 14th ACM
international 2021 WSDM Conference.

Dataset The training dataset consists of over a million of anonymized hotel reservations, based on real data, with the
following features:

- `user_id` - User ID
- `check`-in - Reservation check-in date
- `checkout` - Reservation check-out date
- `affiliate_id` - An anonymized ID of affiliate channels where the booker came from (e.g. direct, some third party
  referrals, paid search engine, etc.)
- `device_class` - desktop/mobile
- `booker_country` - Country from which the reservation was made (anonymized)
- `hotel_country` - Country of the hotel (anonymized)
- `city_id` - city_id of the hotel’s city (anonymized)
- `utrip_id` - Unique identific


Each reservation is a part of a customer’s trip (identified by utrip_id) which includes at least 4 consecutive
reservations. The check-out date of a reservation is the check-in date of the following reservation in their trip.
The evaluation dataset is constructed similarly, however the city_id of the final reservation of each trip is concealed
and requires a prediction.

## Evaluation criteria 

The goal of the challenge is to predict (and recommend) the final city (city_id) of each trip (
utrip_id). We will evaluate the quality of the predictions based on the top four recommended cities for each trip by
using Precision@4 metric (4 representing the four suggestion slots at Booking.com website). When the true city is one of
the top 4 suggestions (regardless of the order), it is considered correct.

## Competition terms and conditions 

The dataset is a property of Booking.com and may not be reused for commercial purposes.
Employees of online travel platform companies or other booking services (including Booking Holdings employees) are not
eligible to compete for prizes in the challenge.
Participants are allowed to participate only once, with no concurrent submissions or code sharing between the teams.
The organizer is authorized to change the prize to award one that’s equivalent in its monetary value.

## Submission guidelines 

The test set will be released to registered e-mails on January 14st, 2021. The teams are expected
to submit their top four city predictions per each trip on the test set until January 28th 2021. The submission should
be completed on easychair website  (https://easychair.org/conferences/?conf=bookingwebtour21). in a csv file named
submission.csv with the following columns;

utrip_id - 1000031_1
city_id_1 - 8655
city_id_2 - 8652
city_id_3 - 4323
city_id_4 - 4332

Where utrip_id represents each unique trip in the test and the rest of the columns represent the city_id of top 4
predicted cities.

On February 4th, 2021 the organizers will reveal the performance on the test set and will announce the final
leaderboard.

The top 10 teams will be invited to submit short papers (up to 4 pages + references in ACM sigconf format). The papers
will include the team and the authors names, an abstract, a text describing the method and the achieved score, and a
link to their code repository. Please refer to the Booking.com WSDM WebTour 21 challenge in the following format:

Dmitri Goldenberg, Kostia Kofman, Pavel Levin, Sarai Mizrachi, Maayan Kafry, and Guy Nadav. 2021. Booking.com WSDM
WebTour 2021 Challenge. https://www.bookingchallenge.com. In ACM WSDM Workshop on Web Tourism (WSDM Webtour’21), March
12, 2021, Jerusalem, Israel.

Bibtex:
@InProceedings{booking2021challenge,
author = {Goldenberg, Dmitri and Kofman, Kostia and Levin, Pavel and Mizrachi, Sarai and Kafry, Maayan and Nadav, Guy},
title = {Booking.com WSDM WebTour 2021 Challenge},
booktitle = {ACM WSDM Workshop on Web Tourism (WSDM WebTour’21)},
year = {2021},
howpublished = {\url{https://www.bookingchallenge.com}},


Paper submission is mandatory in order to be eligible for a prize (top 3 scores and best paper award). Selected papers
are expected to present their work in the workshop (in a virtual format). Please note that the paper quality will be
peer-reviewed. Badly written papers or absence from the workshop may prevent the team from being eligible for a prize.
The submitted papers will be peer-reviewed and evaluated based on their clarity, novelty, and results presentation.

For any problems or questions please contact wsdmchallenge@booking.com

# Data dictionary

## Raw

Raw data as given by Booking

 - `user_id` - User ID
 - `check`-in - Reservation check-in date
 - `checkout` - Reservation check-out date
 - `affiliate_id` - An anonymized ID of affiliate channels where the booker came from (e.g. direct, some third party
    referrals, paid search engine, etc.)
 - `device_class` - desktop/mobile
 - `booker_country` - Country from which the reservation was made (anonymized)
 - `hotel_country` - Country of the hotel (anonymized)
 - `city_id` - city_id of the hotel’s city (anonymized)
 - `utrip_id` - Unique identification of user’s trip (a group of multi-destinations bookings within the same trip)