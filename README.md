
# Warehouse Automation: Reinforcement Learning for Robot Navigation

This project implements **Reinforcement Learning** to train robots to navigate a storage warehouse, avoid obstacles, and deliver packages efficiently. Using algorithms like **Q-Learning** and **SARSA**, the model explores strategies for optimal movement within the **Gymnasium Frozen Lake environment**. The final result is a visual emulation of the trained robot navigating the warehouse, showcasing its learned path.

## 🌍 **Business Value and Real-World Applications**

Warehouse automation is at the forefront of supply chain innovation, driven by increasing consumer demand for faster shipping and the rising trend of intelligent robotics. This project demonstrates how **Reinforcement Learning (RL)** can optimize robot navigation in a warehouse to address real-world challenges and deliver measurable business value.

### **Key Benefits:**
- **Enhanced Productivity**: RL algorithms improve robot navigation, reducing travel time and enabling faster, more accurate package deliveries.  
- **Cost Efficiency**: Automating repetitive tasks reduces labor costs, minimizes operational downtime, and decreases energy consumption.  
- **Customer Satisfaction**: By shortening shipping times and ensuring reliable delivery, companies can meet the expectations of modern consumers who demand fast, often free, shipping.

### **Real-World Applications:**
- **Supply Chain Optimization**: With 75% of large enterprises expected to adopt intelligent robots by 2026 (Gartner, 2022), solutions like this project align with the industry's trajectory.  
- **Scalability in Operations**: Customizable RL models enable adaptation to dynamic warehouse layouts and varied operational requirements.  
- **Driving Innovation**: Investments like Amazon's $1 billion in supply chain automation highlight the growing focus on AI-driven logistics.  


## 🛠️ **Technologies Used**

- Python
- OpenAI Gym & Gymnasium
- Reinforcement Learning Algorithms:
  - Q-Learning
  - SARSA
- Visualization:
  - PyGame
  - IPyWidgets
  - TQDM (progress tracking)

## 🔎 **What the Code Does**

The project is organized into the following key components:

1. **Environment Setup**:  
   - Configures and customizes the Gymnasium Frozen Lake environment to be applicable for warehouse automation training.  
   - Supports custom maps for simulating different warehouse layouts.  

2. **Reinforcement Learning**:  
   - Implements Q-Learning and SARSA algorithms with customizable hyperparameters.  
   - Runs simulations to train agents on navigation strategies.  

3. **Visualization**:  
   - Provides a frame-by-frame emulation of the trained robot navigating the warehouse.  

4. **Evaluation**:  
   - Compares performance metrics between Q-Learning and SARSA for different hyperparameter configurations.  

## 🚀 **How to Run the Code**

1. Clone the repository:  
   ```bash
   git clone https://github.com/NamSingh/Warehouse-Automation.git
   cd Warehouse-Automation
   ```

2. Update the image folder location (ctrl+f "Task 1" to find the task in the code)

3. Modify and update any required hyperparameters for training (ctrl+f "Task 2" to find the task in the code)

4. Customize the project and warehouse (optional):  
   - Create custom maps for Q-Learning (`Task 4`) or SARSA (`Task 5`).

5. Run the code in your preferred format:  
   - For Jupyter Notebook:  
     Open and run `RL-WarehouseAutomation.ipynb`.  
   - For Python script:  
     Execute the Python file to see a frame-by-frame emulation of the robot's movements:  
     ```bash
     python RL-WarehouseAutomation.py
     ```

## 🚀 **Key Highlights**

- **Faster Training with Q-Learning**: Q-Learning demonstrated faster training times and required fewer episodes compared to SARSA, highlighting its efficiency for this application.  
- **Scalability with Deterministic Maps**: Successfully tested custom deterministic maps up to 25 x 25 in size, demonstrating the algorithm's ability to handle complex environments.  
- **Performance on Random Maps**: RL algorithms worked effectively on randomly generated maps up to 15 x 15 in size, with a 10% probability of shelves, showcasing adaptability to dynamic and unpredictable layouts.  
- **Improved Efficiency**: Achieved significant reductions in robot travel time and energy consumption, leading to higher productivity and optimized operations.  
- **Practical Impact**: This solution lays the groundwork for scalable, intelligent automation in warehouse logistics, aligning with industry trends and technological advancements.   

## 📜 **References**

- OpenAI Gym Frozen Lake: [Link](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)