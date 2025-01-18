const { useState, useEffect } = React;

const initialData = {
    "Новые задачи": [],
    "Обработано ботом": [],
    "Передано менеджеру": []
};

function KanbanBoard() {
    const [columns, setColumns] = useState(initialData);
    const [hoveredColumn, setHoveredColumn] = useState(null);
    const [hoveredTaskIndex, setHoveredTaskIndex] = useState(null);
    const [selectedTask, setSelectedTask] = useState(null);

    useEffect(() => {
        fetch('/deals')
            .then(response => response.json())
            .then(data => {
                const newTasks = [];
                const processedByBotTasks = [];
                const handedToManagerTasks = [];

                data.forEach(deal => {
                    switch (deal.stage_name) {
                        case "Новые задачи":
                            newTasks.push(deal);
                            break;
                        case "Обработано ботом":
                            processedByBotTasks.push(deal);
                            break;
                        case "Передано менеджеру":
                            handedToManagerTasks.push(deal);
                            break;
                        default:
                            newTasks.push(deal);
                    }
                });

                setColumns({
                    "Новые задачи": newTasks,
                    "Обработано ботом": processedByBotTasks,
                    "Передано менеджеру": handedToManagerTasks
                });
            })
            .catch(error => console.error('Ошибка при загрузке сделок:', error));
    }, []);

    const handleCreateDeal = () => {
        const newDeal = {
            id: Date.now(),
            name: "Новая сделка",
            price: "0 руб",
            stage_name: "Новые задачи"
        };

        setColumns(prevColumns => ({
            ...prevColumns,
            "Новые задачи": [...prevColumns["Новые задачи"], newDeal]
        }));
    };

    const moveTask = (taskId, fromColumn, toColumn) => {
        const taskToMove = columns[fromColumn].find(task => task.id === taskId);
        const updatedFromColumn = columns[fromColumn].filter(task => task.id !== taskId);
        const updatedToColumn = [...columns[toColumn]];

        if (taskToMove) {
            if (hoveredTaskIndex !== null && hoveredTaskIndex < updatedToColumn.length) {
                updatedToColumn.splice(hoveredTaskIndex, 0, taskToMove);
            } else {
                updatedToColumn.push(taskToMove);
            }

            setColumns({
                ...columns,
                [fromColumn]: updatedFromColumn,
                [toColumn]: updatedToColumn
            });

            setHoveredTaskIndex(null);
        }
    };

    const handleDrop = (e, toColumn) => {
        e.preventDefault();
        const taskId = e.dataTransfer.getData("text/plain");
        const fromColumn = e.dataTransfer.getData("fromColumn");
        if (fromColumn !== toColumn) {
            moveTask(Number(taskId), fromColumn, toColumn);
        }
        setHoveredColumn(null);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleDragEnter = (column, index) => {
        setHoveredColumn(column);
        setHoveredTaskIndex(index);
    };

    const handleDragLeave = () => {
        setHoveredColumn(null);
        setHoveredTaskIndex(null);
    };

    const handleTaskClick = (task) => {
        setSelectedTask(task);
    };

    const closeTaskDetails = () => {
        setSelectedTask(null);
    };

    const updateTaskName = (taskId, newName) => {
        setColumns((prevColumns) => {
            const updatedColumns = { ...prevColumns };
            for (const column in updatedColumns) {
                const taskIndex = updatedColumns[column].findIndex(task => task.id === taskId);
                if (taskIndex !== -1) {
                    updatedColumns[column][taskIndex].name = newName;
                    break;
                }
            }
            return updatedColumns;
        });

        // Обновляем выбранную задачу, если она редактируется
        setSelectedTask((prevTask) => {
            if (prevTask && prevTask.id === taskId) {
                return { ...prevTask, name: newName };
            }
            return prevTask;
        });
    };

    return (
    <div className="kanban-board-container">
        <div className="kanban-header">
            {/* Добавляем кнопку над колонками справа */}
            <button onClick={handleCreateDeal} className="create-deal-button">Добавить</button>
        </div>

        {/* Контейнер для колонок */}
        <div className="kanban-board">
            {Object.keys(columns).map((column) => (
                <Column
                    key={column}
                    title={column}
                    tasks={columns[column]}
                    handleDrop={handleDrop}
                    handleDragOver={handleDragOver}
                    handleDragEnter={handleDragEnter}
                    handleDragLeave={handleDragLeave}
                    hoveredTaskIndex={hoveredTaskIndex}
                    hoveredColumn={hoveredColumn}
                    onTaskClick={handleTaskClick}
                />
            ))}
        </div>

        {/* Дополнительный интерфейс */}
        {hoveredColumn && hoveredTaskIndex !== null && (
            <div className="drop-shadow" />
        )}
        {selectedTask && (
            <TaskDetails
                task={selectedTask}
                onClose={closeTaskDetails}
                updateTaskName={updateTaskName}
            />
        )}
    </div>
);
}

function Column({ title, tasks, handleDrop, handleDragOver, handleDragEnter, handleDragLeave, hoveredTaskIndex, hoveredColumn, onTaskClick }) {
    const calculateTotal = (tasks) => {
        return tasks.reduce((total, task) => {
            const price = parseInt(task.price.replace(/\s/g, '').replace('руб', ''));
            return total + price;
        }, 0);
    };

    const totalSum = calculateTotal(tasks);

    return (
        <div
            className="column"
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, title)}
            onDragLeave={handleDragLeave}
        >
            <div className="summary-container">
                <div className="column-circle">{tasks.length}</div>
                <span className="column-title">{title}</span>
            </div>
            <div className="column-sum">{totalSum.toLocaleString('ru-RU')} руб</div>
            <div className="tasks">
                {tasks.map((task, index) => (
                    <Task
                        key={task.id}
                        task={task}
                        handleDragStart={(e) => {
                            e.dataTransfer.setData("text/plain", task.id);
                            e.dataTransfer.setData("fromColumn", title);
                        }}
                        isHovered={hoveredColumn === title && hoveredTaskIndex === index}
                        onClick={() => onTaskClick(task)}
                    />
                ))}
            </div>
        </div>
    );
}


function TaskDetails({ task, onClose, updateTaskName }) {
    const [taskName, setTaskName] = useState(task.name);
    const [isEditing, setIsEditing] = useState(false);

    const handleNameChange = (e) => {
        const newName = e.target.innerText;
        setTaskName(newName);
        updateTaskName(task.id, newName);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            setIsEditing(false);
        }
    };

    const handleFocus = (e) => {
        if (!isEditing) {
            setIsEditing(true);
        }
    };

    return (
        <div className="task-details">
            <div className="task-details-container">
                <button onClick={onClose}>Закрыть</button>
                <h3 className="task-details-title task-num">#{task.id}</h3>
                <div>
                    {isEditing ? (
                        <span
                            onBlur={() => setIsEditing(false)}
                            onKeyDown={handleKeyDown}
                            onInput={handleNameChange}
                            contentEditable
                            suppressContentEditableWarning={true}
                            className="editable-title"
                            onClick={(e) => e.stopPropagation()}
                            style={{ border: 'none', outline: 'none' }}
                        >
                            {taskName}
                        </span>
                    ) : (
                        <span
                            onClick={() => setIsEditing(true)}
                            className="editable-title"
                        >
                            {taskName}
                        </span>
                    )}
                </div>
                <h2 className="task-details-price"><strong></strong> {task.price}</h2>

                {/* Выровненные поля ввода */}
                <div className="input-group">
                    <label htmlFor="lastName" className="input-label">Фамилия</label>
                    <input id="lastName" type="text" className="input-field" placeholder="Введите фамилию" />
                </div>
                <div className="input-group">
                    <label htmlFor="Name" className="input-label">Имя</label>
                    <input id="Name" type="text" className="input-field" placeholder="Введите имя" />
                </div>
                <div className="input-group">
                    <label htmlFor="patronymic" className="input-label">Отчество</label>
                    <input id="patronymic" type="text" className="input-field" placeholder="Введите отчество" />
                </div>
                <div className="input-group">
                    <label htmlFor="phoneNumber" className="input-label">Номер телефона</label>
                    <input id="phoneNumber" type="text" className="input-field" placeholder="Введите номер телефона" />
                </div>
                <div className="input-group">
                    <label htmlFor="email" className="input-label">Электронная почта</label>
                    <input id="email" type="email" className="input-field" placeholder="Введите электронную почту" />
                </div>
                <div className="input-group">
                    <label htmlFor="inn" className="input-label">ИНН</label>
                    <input id="inn" type="text" className="input-field" placeholder="Введите ИНН" />
                </div>
                <div className="input-group">
                    <label htmlFor="orderDate" className="input-label">Дата заказа</label>
                    <input id="orderDate" type="date" className="input-field" />
                </div>
            </div>
        </div>
    );
}



function Task({ task, handleDragStart, isHovered, onClick }) {
    return (
        <div
            className={`task ${isHovered ? "hovered" : ""}`}
            draggable
            onDragStart={handleDragStart}
            onClick={onClick}
        >
            <div className="task-content">
                <span className="task-id">#{task.id}</span>
                <span className="task-name">{task.name}</span>
                <div className="price-container">
                    <div className="price">{task.price}</div>
                    <img className="user-icon" src="static/Image/User-photo.png" alt="User" />
                </div>
            </div>
        </div>
    );
}

ReactDOM.render(<KanbanBoard />, document.getElementById('kanban-board'));
