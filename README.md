# opensea-rest
A pipeline to automate data extraction from opensea.io REST API and then model the asset prices using XGBoost or other frameworks. The data scope is Cryptovoxel virtual homes.

![png](https://github.com/datavizhokie/opensea-rest/blob/main/opensea.png)

https://opensea.io/

## What are we actually talking about with Cryptovoxels?

*"The browser-based game allows you to explore a sprawling three dimensional world built with chunky, pixel-like blocks commonly referred to as "voxels." Minecraft is the most famous voxel game, and its success popularized a whole genre of blocky free-play worlds. Like Minecraft, Cryptovoxels lets you explore and build without any strict goal or directive. Where Cryptovoxels differs from a game like Minecraft is in its integration of— you guessed it— crypto! While anyone can explore the Cryptovoxel world freely, to build in the world you have to own property. Ownership is tracked through NFTs on the Ethereum network."*

This is an excerpt from an apt summary of the Cryptovoxel community: https://www.buildblockchain.tech/newsletter/issues/no-101-the-weird-world-of-cryptovoxels


## Data snapshot:

<img src="https://github.com/datavizhokie/opensea-rest/blob/main/tableau_viz.png" width="600" height="525">

## Data considerations

Below are the features extracted from the raw JSON data:

*'cv_plotSize_m_sq':  Plot size in square meters <br/>
*'cv_OCdistance_m':   Distance from the Origin <br/>
*'cv_buildHeight_m':  Build height in meters <br/>
*'cv_floor_elev_m':   Base floor elevation in meters <br/>
*'neighborhood':      Text neighborhood name <br/>
*'near_to':           Sub neighborhoods that location is near <br/>

'Neighborhood' and 'Near To" are categorical fields and must be encoded for XGBoost to consume. The following encoding methods are explored:

* One Hot Encoding (a new field for every attribute of the field) <br/>
* Categorical Encoding (field attributes are converted to unique numeric values) <br/>

Tableau viz of training data: https://public.tableau.com/profile/matt.wheeler#!/vizhome/CryptovoxelData/CryptovoxelVirtualHomes

## Modeling with XGBoost

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. I am using a XGBoost regressor to predict cryptovoxol home sales using features such as *Plot Size* and *Distance from Origin City*.

Intial tuning jobs yield the following results, comparing the Neighborhood field as One Hot Encoded and as Categorical Encoded:

<img src="https://github.com/datavizhokie/opensea-rest/blob/main/xgb_initial_1he.png" width="450" height="400">
<img src="https://github.com/datavizhokie/opensea-rest/blob/main/xgb_initial_cat_encode.png" width="450" height="400">
